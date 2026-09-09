package nca

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
)

const (
	magic     = "NCA1"
	version   = 1
	flagTanh  = 1 << 0
	flagClamp = 1 << 1
)

// update rule: a 3x3 convolution
// then two-layer 1x1 MLP
type Model struct {
	Channels int
	Percept  int
	Kernel   int
	Hidden   int

	FireRate      float32
	Noise         float32
	LifeThreshold float32
	LifeChannel   int
	tanh          bool
	clamp         bool

	perceptW []float32 // [Percept][Channels][Kernel][Kernel]
	perceptB []float32 // [Percept]
	hiddenW  []float32 // [Hidden][Percept]
	hiddenB  []float32 // [Hidden]
	outW     []float32 // [Channels][Hidden]
	outB     []float32 // [Channels]

	bands   []*band
	alive   []bool
	preLife []bool
}

type band struct {
	percept []float32
	hidden  []float32
	delta   []float32
	neigh   []float32
	rng     *rand.Rand
}

// Reads a weights file exported by training/export_weights.py.
func LoadFile(path string) (*Model, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return Load(bytes.NewReader(data))
}

// Reads the weights format documented in training/export_weights.py.
func Load(r io.Reader) (*Model, error) {
	var header struct {
		Magic         [4]byte
		Version       uint32
		Channels      uint32
		Percept       uint32
		Kernel        uint32
		Hidden        uint32
		FireRate      float32
		Noise         float32
		LifeThreshold float32
		LifeChannel   uint32
		Flags         uint32
	}
	if err := binary.Read(r, binary.LittleEndian, &header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header.Magic[:]) != magic {
		return nil, fmt.Errorf("not an nca file (magic %q)", header.Magic)
	}
	if header.Version != version {
		return nil, fmt.Errorf("unsupported nca version %d", header.Version)
	}

	m := &Model{
		Channels:      int(header.Channels),
		Percept:       int(header.Percept),
		Kernel:        int(header.Kernel),
		Hidden:        int(header.Hidden),
		FireRate:      header.FireRate,
		Noise:         header.Noise,
		LifeThreshold: header.LifeThreshold,
		LifeChannel:   int(header.LifeChannel),
		tanh:          header.Flags&flagTanh != 0,
		clamp:         header.Flags&flagClamp != 0,
	}
	if m.Kernel != 3 {
		return nil, fmt.Errorf("kernel %d unsupported, forward pass assumes 3", m.Kernel)
	}

	for _, section := range []struct {
		dst *[]float32
		n   int
	}{
		{&m.perceptW, m.Percept * m.Channels * m.Kernel * m.Kernel},
		{&m.perceptB, m.Percept},
		{&m.hiddenW, m.Hidden * m.Percept},
		{&m.hiddenB, m.Hidden},
		{&m.outW, m.Channels * m.Hidden},
		{&m.outB, m.Channels},
	} {
		buf := make([]float32, section.n)
		if err := binary.Read(r, binary.LittleEndian, buf); err != nil {
			return nil, fmt.Errorf("read weights: %w", err)
		}
		*section.dst = buf
	}

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		m.bands = append(m.bands, &band{
			percept: make([]float32, m.Percept),
			hidden:  make([]float32, m.Hidden),
			delta:   make([]float32, m.Channels),
			neigh:   make([]float32, m.Channels*9),
		})
	}
	return m, nil
}

// Writes the next state of the grid into next.
//
// The stages, in the order the trained model expects them:
//
//	noise: gaussian jitter added before perception, so the rule spends training correcting real drift and stays stable when idling
//	perceive: 3x3 conv with wrap-around padding, then leaky ReLU
//	update: 1x1 MLP -> per-channel delta
//	fire rate: each cell ignores the delta with probability 1-FireRate, which is what keeps neighbours from updating in lockstep
//	life mask: a cell is alive only if it was alive before and after the update. Dead cells are zeroed outright, which is what stops the creature from smearing into the empty grid
//	clamp: state held in [-1, 1]
func (m *Model) Step(state, next []float32, width, height int, rng *rand.Rand) {
	cells := width * height
	if len(state) != m.Channels*cells || len(next) != len(state) {
		panic("nca: state size does not match model")
	}
	if cap(m.alive) < cells {
		m.alive = make([]bool, cells)
		m.preLife = make([]bool, cells)
	}
	alive, preLife := m.alive[:cells], m.preLife[:cells]

	for _, b := range m.bands {
		b.rng = rand.New(rand.NewSource(rng.Int63()))
	}

	m.livingMask(state, preLife, width, height)

	if m.Noise > 0 {
		for i := range state {
			state[i] += float32(rng.NormFloat64()) * m.Noise
		}
	}

	copy(next, state)
	m.eachBand(height, func(b *band, from, to int) {
		for y := from; y < to; y++ {
			for x := 0; x < width; x++ {
				// Cells with a dead neighbourhood are zeroed by the living mask
				if !preLife[y*width+x] {
					continue
				}
				// Fire rate: a cell that does not fire keeps its old value.
				if m.FireRate < 1 && b.rng.Float32() > m.FireRate {
					continue
				}
				m.perceive(b, state, width, height, x, y)
				m.update(b)
				for c, d := range b.delta {
					next[(c*height+y)*width+x] += d
				}
			}
		}
	})

	m.livingMask(next, alive, width, height)
	for cell := 0; cell < cells; cell++ {
		live := alive[cell] && preLife[cell]
		for c := 0; c < m.Channels; c++ {
			i := c*cells + cell
			switch {
			case !live:
				next[i] = 0
			case m.clamp:
				next[i] = clamp1(next[i])
			}
		}
	}
}

// perceive fills m.percept with the convolution over the cell's neighbourhood.
// Padding is circular
func (m *Model) perceive(b *band, state []float32, width, height, x, y int) {
	cells := width * height

	neigh := b.neigh
	for ky := -1; ky <= 1; ky++ {
		row := wrap(y+ky, height) * width
		for kx := -1; kx <= 1; kx++ {
			at := row + wrap(x+kx, width)
			k := (ky+1)*3 + (kx + 1)
			for c := 0; c < m.Channels; c++ {
				neigh[c*9+k] = state[c*cells+at]
			}
		}
	}

	n := len(neigh)
	for p := range b.percept {
		sum := m.perceptB[p] + dot(m.perceptW[p*n:(p+1)*n], neigh)
		b.percept[p] = leakyReLU(sum)
	}
}

// update runs the 1x1 MLP over the perceived features
func (m *Model) update(b *band) {
	for h := range b.hidden {
		row := h * m.Percept
		sum := m.hiddenB[h] + dot(m.hiddenW[row:row+m.Percept], b.percept)
		b.hidden[h] = leakyReLU(sum)
	}
	for c := 0; c < m.Channels; c++ {
		row := c * m.Hidden
		sum := m.outB[c] + dot(m.outW[row:row+m.Hidden], b.hidden)
		if m.tanh {
			sum = float32(math.Tanh(float64(sum)))
		}
		b.delta[c] = sum
	}
}

// Marks a cell alive when any alpha in its 3x3 neighbourhood is above the threshold
func (m *Model) livingMask(state []float32, out []bool, width, height int) {
	alpha := state[m.LifeChannel*width*height:]
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			strongest := float32(math.Inf(-1))
			for ky := -1; ky <= 1; ky++ {
				yy := y + ky
				if yy < 0 || yy >= height {
					continue // max_pool2d pads with zeros, i.e. never wins
				}
				for kx := -1; kx <= 1; kx++ {
					xx := x + kx
					if xx < 0 || xx >= width {
						continue
					}
					if v := alpha[yy*width+xx]; v > strongest {
						strongest = v
					}
				}
			}
			out[y*width+x] = strongest > m.LifeThreshold
		}
	}
}

func (m *Model) eachBand(height int, work func(b *band, from, to int)) {
	bands := len(m.bands)
	if bands > height {
		bands = height
	}
	var wg sync.WaitGroup
	for i := 0; i < bands; i++ {
		from, to := i*height/bands, (i+1)*height/bands
		wg.Add(1)
		go func(b *band, from, to int) {
			defer wg.Done()
			work(b, from, to)
		}(m.bands[i], from, to)
	}
	wg.Wait()
}

// dot sums w[i]*v[i]
func dot(w, v []float32) float32 {
	if len(w) > len(v) {
		w = w[:len(v)]
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= len(w); i += 4 {
		s0 += w[i] * v[i]
		s1 += w[i+1] * v[i+1]
		s2 += w[i+2] * v[i+2]
		s3 += w[i+3] * v[i+3]
	}
	sum := (s0 + s1) + (s2 + s3)
	for ; i < len(w); i++ {
		sum += w[i] * v[i]
	}
	return sum
}

func wrap(i, n int) int {
	if i < 0 {
		return i + n
	}
	if i >= n {
		return i - n
	}
	return i
}

func leakyReLU(v float32) float32 {
	if v < 0 {
		return v * 0.2
	}
	return v
}

func clamp1(v float32) float32 {
	if v < -1 {
		return -1
	}
	if v > 1 {
		return 1
	}
	return v
}
