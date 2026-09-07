// Command tardigrade grows a self-healing creature in your terminal.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/gdamore/tcell/v2"

	"github.com/smoothyy3/tardigrade/assets"
	"github.com/smoothyy3/tardigrade/internal/grid"
	"github.com/smoothyy3/tardigrade/internal/input"
	"github.com/smoothyy3/tardigrade/internal/nca"
	"github.com/smoothyy3/tardigrade/internal/render"
	"github.com/smoothyy3/tardigrade/internal/supervisor"
)

const (
	// keyWound is what pressing `d` costs the creature, in cells.
	keyWound = 14

	screensaverEvery = 6 * time.Second
	screensaverFPS   = 10
)

type options struct {
	weights     string
	size        int
	fps         int
	seed        int64
	screensaver bool
}

func main() {
	var opts options
	flag.StringVar(&opts.weights, "weights", "", "path to a weights file (default: the ones built into the binary)")
	flag.IntVar(&opts.size, "size", 56, "grid size in cells (the creature was trained at 56)")
	flag.IntVar(&opts.fps, "fps", 20, "frames per second")
	flag.Int64Var(&opts.seed, "seed", -1, "random seed (-1: a new one every run)")
	flag.BoolVar(&opts.screensaver, "screensaver", false, "wound and heal on a timer, exit on any key")
	worker := flag.Bool("worker", false, "run the creature itself (the supervisor sets this)")
	flag.Parse()

	// screensaver idles at half speed
	if opts.screensaver && !isSet("fps") {
		opts.fps = screensaverFPS
	}

	var err error
	if *worker {
		err = runWorker(opts)
	} else {
		err = supervisor.Run(workerFlags(opts))
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "tardigrade:", err)
		os.Exit(1)
	}
}

func isSet(name string) bool {
	seen := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == name {
			seen = true
		}
	})
	return seen
}

func workerFlags(opts options) []string {
	args := []string{
		"--size", strconv.Itoa(opts.size),
		"--fps", strconv.Itoa(opts.fps),
	}

	if opts.seed >= 0 {
		args = append(args, "--seed", strconv.FormatInt(opts.seed, 10))
	}
	if opts.weights != "" {
		args = append(args, "--weights", opts.weights)
	}
	if opts.screensaver {
		args = append(args, "--screensaver")
	}
	return args
}

func runWorker(opts options) error {
	fmt.Printf("tardigrade %d\n", os.Getpid())

	model, err := loadModel(opts.weights)
	if err != nil {
		return err
	}
	seed := opts.seed
	if seed < 0 {
		seed = time.Now().UnixNano()
	}
	g := grid.New(model, opts.size, opts.size, seed)
	rng := rand.New(rand.NewSource(seed))

	screen, err := tcell.NewScreen()
	if err != nil {
		return err
	}
	if err := screen.Init(); err != nil {
		return err
	}
	defer screen.Fini()
	screen.HideCursor()
	if !opts.screensaver {
		screen.EnableMouse(tcell.MouseButtonEvents | tcell.MouseDragEvents)
	}

	status := fmt.Sprintf("pid %d  ·  drag to wound  ·  d: bite  ·  r: regrow  ·  q: quit", os.Getpid())
	if opts.screensaver {
		status = fmt.Sprintf("pid %d  ·  any key to exit", os.Getpid())
	}

	events := make(chan tcell.Event, 8)
	go func() {
		for {
			events <- screen.PollEvent()
		}
	}()

	frame := time.NewTicker(time.Second / time.Duration(opts.fps))
	defer frame.Stop()
	timer := time.NewTicker(screensaverEvery)
	defer timer.Stop()

	for {
		select {
		case event := <-events:
			if _, isKey := event.(*tcell.EventKey); isKey && opts.screensaver {
				return nil
			}
			originX, originY := render.Origin(screen, g)
			switch action := input.Interpret(event, originX, originY); action.Action {
			case input.Quit:
				return nil
			case input.Redraw:
				screen.Clear()
				screen.Sync()
			case input.WoundAt:
				g.ApplyDamage(action.Wound)
			case input.WoundRandom:
				g.ApplyDamage(randomWound(rng, g, keyWound))
			case input.Regrow:
				g.Seed()
			}

		case <-timer.C:
			if opts.screensaver {
				g.ApplyDamage(randomWound(rng, g, keyWound))
			}

		case <-frame.C:
			if g.Empty() {
				g.Seed()
			}
			g.Step()
			render.Draw(screen, g)
			render.Status(screen, status)
			screen.Show()
		}
	}
}

func loadModel(path string) (*nca.Model, error) {
	if path != "" {
		return nca.LoadFile(path)
	}
	return nca.Load(bytes.NewReader(assets.Tardigrade))
}

func randomWound(rng *rand.Rand, g *grid.Grid, size int) grid.Rect {
	return grid.Rect{
		X: rng.Intn(max(g.Width-size, 1)) - size/4,
		Y: rng.Intn(max(g.Height-size, 1)) - size/4,
		W: size,
		H: size,
	}
}
