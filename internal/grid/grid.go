package grid

import (
	"math/rand"

	"github.com/smoothyy3/tardigrade/internal/nca"
)

const (
	alphaChannel = 3

	// AliveAlpha = CA living threshold
	AliveAlpha = 0.1

	// At this point not a creature anymore but "all alive" dead end
	saturatedAlive = 0.9
)

// Region of cells, in cell coordinates.
type Rect struct{ X, Y, W, H int }

// Grid = creature
type Grid struct {
	Width, Height int

	model       *nca.Model
	state, next []float32
	rng         *rand.Rand
}

// Returns a seeded grid ready to grow.
func New(model *nca.Model, width, height int, seed int64) *Grid {
	g := &Grid{
		Width:  width,
		Height: height,
		model:  model,
		state:  make([]float32, model.Channels*width*height),
		next:   make([]float32, model.Channels*width*height),
		rng:    rand.New(rand.NewSource(seed)),
	}
	g.Seed()
	return g
}

// Seed resets
func (g *Grid) Seed() {
	clear(g.state)
	centre := (g.Height/2)*g.Width + g.Width/2
	for c := 0; c < 4; c++ {
		g.state[c*g.Width*g.Height+centre] = 1
	}
}

// Advances the automaton one step
func (g *Grid) Step() {
	g.model.Step(g.state, g.next, g.Width, g.Height, g.rng)
	g.state, g.next = g.next, g.state
}

// Zeroes every channel of every cell in r
func (g *Grid) ApplyDamage(r Rect) {
	x0, y0 := max(r.X, 0), max(r.Y, 0)
	x1, y1 := min(r.X+r.W, g.Width), min(r.Y+r.H, g.Height)
	cells := g.Width * g.Height
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			for c := 0; c < g.model.Channels; c++ {
				g.state[c*cells+y*g.Width+x] = 0
			}
		}
	}
}

// Empty if nothing is alive. Dead end.
func (g *Grid) Empty() bool {
	alpha := g.state[alphaChannel*g.Width*g.Height:][:g.Width*g.Height]
	for _, a := range alpha {
		if a > AliveAlpha {
			return false
		}
	}
	return true
}

// If creature has filled whole canvas (dead end)
func (g *Grid) Saturated() bool {
	alpha := g.state[alphaChannel*g.Width*g.Height:][:g.Width*g.Height]
	live := 0
	for _, a := range alpha {
		if a > AliveAlpha {
			live++
		}
	}
	return float64(live) >= saturatedAlive*float64(len(alpha))
}

// Returns a cell's visible colour and alpha
func (g *Grid) Pixel(x, y int) (r, gr, b, a float32) {
	cells := g.Width * g.Height
	at := y*g.Width + x
	a = clamp01(g.state[alphaChannel*cells+at])
	if a < AliveAlpha {
		return 0, 0, 0, a
	}
	return clamp01(g.state[at] / a),
		clamp01(g.state[cells+at] / a),
		clamp01(g.state[2*cells+at] / a),
		a
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}
