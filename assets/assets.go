// Package assets carries the trained weights inside the binary, so an
// installed tardigrade runs from anywhere with no files beside it.
package assets

import (
	_ "embed"
	"fmt"
	"sort"
	"strings"
)

// Each file is exported by training/export_weights.py, in the format documented there and read by internal/nca.
//
//go:embed tardigrade.nca
var tardigrade []byte

//go:embed gecko.nca
var gecko []byte

//go:embed jellyfish.nca
var jellyfish []byte

//go:embed mouse.nca
var mouse []byte

var creatures = map[string][]byte{
	"tardigrade": tardigrade,
	"gecko":      gecko,
	"jellyfish":  jellyfish,
	"mouse":      mouse,
}

// Default creature
const Default = "tardigrade"

// Weights returns the embedded weights for a named creature.
func Weights(name string) ([]byte, error) {
	if w, ok := creatures[name]; ok {
		return w, nil
	}
	return nil, fmt.Errorf("unknown creature %q (have: %s)", name, strings.Join(Names(), ", "))
}

// Lists the embedded creatures, sorted.
func Names() []string {
	names := make([]string, 0, len(creatures))
	for n := range creatures {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
