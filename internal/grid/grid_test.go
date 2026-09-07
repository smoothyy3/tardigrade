package grid

import (
	"testing"

	"github.com/smoothyy3/tardigrade/internal/nca"
)

func testModel(channels int) *nca.Model {
	return &nca.Model{Channels: channels, Percept: 1, Kernel: 3, Hidden: 1, LifeChannel: 3}
}

func TestSeedIsASingleLiveCell(t *testing.T) {
	g := New(testModel(16), 8, 6, 1)
	live := 0
	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			if _, _, _, a := cell(g, x, y); a != 0 {
				live++
			}
		}
	}
	if live != 1 {
		t.Errorf("seeded grid has %d live cells, want 1", live)
	}
	if _, _, _, a := cell(g, g.Width/2, g.Height/2); a != 1 {
		t.Errorf("centre cell alpha = %v, want 1", a)
	}
}

func TestApplyDamageClearsRectAndClipsToGrid(t *testing.T) {
	g := New(testModel(16), 8, 6, 1)
	for i := range g.state {
		g.state[i] = 1
	}

	g.ApplyDamage(Rect{X: -2, Y: -2, W: 4, H: 4})

	for y := 0; y < g.Height; y++ {
		for x := 0; x < g.Width; x++ {
			_, _, _, a := cell(g, x, y)
			inside := x < 2 && y < 2
			if inside && a != 0 {
				t.Fatalf("cell (%d,%d) inside damage has alpha %v", x, y, a)
			}
			if !inside && a != 1 {
				t.Fatalf("cell (%d,%d) outside damage has alpha %v", x, y, a)
			}
		}
	}
}

func cell(g *Grid, x, y int) (r, gr, b, a float32) {
	cells := g.Width * g.Height
	at := y*g.Width + x
	return g.state[at], g.state[cells+at], g.state[2*cells+at], g.state[3*cells+at]
}

func TestEmptyReportsAWipedGrid(t *testing.T) {
	g := New(testModel(16), 8, 6, 1)
	if g.Empty() {
		t.Error("a seeded grid is not empty")
	}

	g.ApplyDamage(Rect{X: 0, Y: 0, W: g.Width, H: g.Height})
	if !g.Empty() {
		t.Error("a grid damaged everywhere should read as empty")
	}

	g.Seed()
	if g.Empty() {
		t.Error("seeding should bring the grid back to life")
	}
}
