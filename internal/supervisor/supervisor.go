package supervisor

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"time"
)

const (
	minLifetime   = 2 * time.Second
	maxQuickDeath = 3
)

// Spawns the worker and restarts it until it exits of its own accord.
func Run(workerArgs []string) error {
	executable, err := os.Executable()
	if err != nil {
		return err
	}

	quickDeaths := 0
	for {
		// --worker comes first so the worker is greppable as "tardigrade --worker": the demo, and anyone reaching for kill -9, needs to tell it apart from the supervisor at a glance.
		args := append([]string{"--worker"}, workerArgs...)

		worker := exec.Command(executable, args...)
		worker.Stdin, worker.Stdout, worker.Stderr = os.Stdin, os.Stdout, os.Stderr

		started := time.Now()
		err := worker.Run()
		if err == nil {
			return nil
		}

		var exited *exec.ExitError
		if !errors.As(err, &exited) {
			return fmt.Errorf("spawn worker: %w", err)
		}

		if time.Since(started) < minLifetime {
			quickDeaths++
			if quickDeaths >= maxQuickDeath {
				return fmt.Errorf("worker keeps dying immediately: %w", err)
			}
		} else {
			quickDeaths = 0
		}
	}
}
