package render

import (
	"github.com/gdamore/tcell/v2"

	"github.com/smoothyy3/tardigrade/internal/grid"
)

func Draw(screen tcell.Screen, g *grid.Grid) {
	originX, originY := Origin(screen, g)

	for y := 0; y < g.Height; y += 2 {
		for x := 0; x < g.Width; x++ {
			top, topAlive := colorAt(g, x, y)
			bottom, bottomAlive := colorAt(g, x, y+1)

			style := tcell.StyleDefault
			glyph := ' '
			switch {
			case topAlive && bottomAlive:
				glyph, style = '▀', style.Foreground(top).Background(bottom)
			case topAlive:
				glyph, style = '▀', style.Foreground(top)
			case bottomAlive:
				glyph, style = '▄', style.Foreground(bottom)
			}
			screen.SetContent(originX+x, originY+y/2, glyph, nil, style)
		}
	}
}

func Origin(screen tcell.Screen, g *grid.Grid) (x, y int) {
	cols, rows := screen.Size()
	return (cols - g.Width) / 2, (rows - (g.Height+1)/2) / 2
}

func colorAt(g *grid.Grid, x, y int) (tcell.Color, bool) {
	if y >= g.Height {
		return tcell.ColorDefault, false
	}
	r, gr, b, a := g.Pixel(x, y)
	if a < grid.AliveAlpha {
		return tcell.ColorDefault, false
	}
	return tcell.NewRGBColor(int32(r*255), int32(gr*255), int32(b*255)), true
}

func Status(screen tcell.Screen, text string) {
	_, rows := screen.Size()
	style := tcell.StyleDefault.Foreground(tcell.ColorGray)
	for i, r := range []rune(text) {
		screen.SetContent(1+i, rows-1, r, nil, style)
	}
}
