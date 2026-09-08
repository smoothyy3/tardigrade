package grid_test

import (
	"bytes"
	"testing"

	"github.com/smoothyy3/tardigrade/assets"
	"github.com/smoothyy3/tardigrade/internal/grid"
	"github.com/smoothyy3/tardigrade/internal/nca"
)

func BenchmarkStep(b *testing.B) {
	data, err := assets.Weights(assets.Default)
	if err != nil {
		b.Fatal(err)
	}
	model, err := nca.Load(bytes.NewReader(data))
	if err != nil {
		b.Fatal(err)
	}
	g := grid.New(model, 56, 56, 1)
	for i := 0; i < 200; i++ {
		g.Step()
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		g.Step()
	}
}
