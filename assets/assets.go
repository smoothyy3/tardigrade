// Package assets carries the trained weights inside the binary, so an
// installed tardigrade runs from anywhere with no files beside it.
package assets

import _ "embed"

// Tardigrade is the weights file exported by training/export_weights.py, in
// the format documented there and read by internal/nca.
//
//go:embed tardigrade.nca
var Tardigrade []byte
