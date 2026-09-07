package input

import (
	"github.com/gdamore/tcell/v2"

	"github.com/smoothyy3/tardigrade/internal/grid"
)

// Width of mouse drag wound
const woundSize = 4

type Action int

const (
	Ignore Action = iota
	Quit
	Redraw
	WoundAt
	WoundRandom
	Regrow
)

type Event struct {
	Action Action
	Wound  grid.Rect
}

func Interpret(event tcell.Event, originX, originY int) Event {
	switch event := event.(type) {
	case *tcell.EventResize:
		return Event{Action: Redraw}

	case *tcell.EventKey:
		switch {
		case event.Key() == tcell.KeyEscape, event.Key() == tcell.KeyCtrlC, event.Rune() == 'q':
			return Event{Action: Quit}
		case event.Rune() == 'd':
			return Event{Action: WoundRandom}
		case event.Rune() == 'r':
			return Event{Action: Regrow}
		}

	case *tcell.EventMouse:
		if event.Buttons()&tcell.Button1 == 0 {
			return Event{Action: Ignore}
		}
		col, row := event.Position()
		return Event{Action: WoundAt, Wound: grid.Rect{
			X: col - originX - woundSize/2,
			Y: (row-originY)*2 - woundSize/2,
			W: woundSize,
			H: woundSize,
		}}
	}
	return Event{Action: Ignore}
}
