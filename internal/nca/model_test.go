package nca

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"testing"
)

// Pin Go forward pass to trained PyTorch one
func TestForwardMatchesPython(t *testing.T) {
	model, err := LoadFile("../../assets/tardigrade.nca")
	if err != nil {
		t.Fatalf("load weights: %v", err)
	}
	input, want, width, height := readFixture(t, "testdata/forward_case.bin")
	if len(input) != model.Channels*width*height {
		t.Fatalf("fixture has %d values, model expects %d channels", len(input), model.Channels)
	}

	model.Noise = 0
	model.FireRate = 1

	got := make([]float32, len(input))
	model.Step(input, got, width, height, rand.New(rand.NewSource(1)))

	const tolerance = 1e-4
	worst := 0.0
	for i := range want {
		if d := math.Abs(float64(want[i] - got[i])); d > worst {
			worst = d
		}
	}
	if worst > tolerance {
		t.Errorf("forward pass diverges from PyTorch: worst |delta| = %g", worst)
	}
}

func readFixture(t *testing.T, path string) (input, output []float32, width, height int) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	f := bytes.NewReader(data)

	var header struct {
		Magic         [4]byte
		Version       uint32
		Channels      uint32
		Height, Width uint32
	}
	if err := binary.Read(f, binary.LittleEndian, &header); err != nil {
		t.Fatalf("read fixture header: %v", err)
	}
	if string(header.Magic[:]) != "NCAF" {
		t.Fatalf("not a fixture file (magic %q)", header.Magic)
	}

	n := int(header.Channels * header.Height * header.Width)
	input, output = make([]float32, n), make([]float32, n)
	if err := binary.Read(f, binary.LittleEndian, input); err != nil {
		t.Fatalf("read fixture input: %v", err)
	}
	if err := binary.Read(f, binary.LittleEndian, output); err != nil {
		t.Fatalf("read fixture output: %v", err)
	}
	return input, output, int(header.Width), int(header.Height)
}

// Benchmark grown creature
func BenchmarkStep(b *testing.B) {
	const size = 56
	m, err := LoadFile("../../assets/tardigrade.nca")
	if err != nil {
		b.Fatalf("load weights: %v", err)
	}
	state := make([]float32, m.Channels*size*size)
	next := make([]float32, len(state))
	for c := 0; c < 4; c++ {
		state[c*size*size+(size/2)*size+size/2] = 1
	}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 120; i++ {
		m.Step(state, next, size, size, rng)
		state, next = next, state
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Step(state, next, size, size, rng)
		state, next = next, state
	}
}
