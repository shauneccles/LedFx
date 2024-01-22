import logging

import numpy as np
import voluptuous as vol

from ledfx.effects.twod import Twod

_LOGGER = logging.getLogger(__name__)


# copy this file and rename it into the effects folder
# Anywhere you see template, replace it with your own class reference / name
class Particle:
    def __init__(self, position, velocity, color, intensity):
        self.position = position
        self.velocity = velocity
        self.color = color
        self.intensity = intensity


class SpectralPaint(Twod):
    NAME = "Spectral Painting"
    CATEGORY = "Matrix"
    # add keys you want hidden or in advanced here
    HIDDEN_KEYS = Twod.HIDDEN_KEYS + []
    ADVANCED_KEYS = Twod.ADVANCED_KEYS + []

    CONFIG_SCHEMA = vol.Schema(
        {
            vol.Optional(
                "a_switch",
                description="Does a boolean thing",
                default=False,
            ): bool,
        }
    )

    def __init__(self, ledfx, config):
        super().__init__(ledfx, config)
        # Initialize intensity attributes
        self.lows_intensity = 0
        self.mids_intensity = 0
        self.highs_intensity = 0

    def config_updated(self, config):
        super().config_updated(config)
        # copy over your configs here into variables
        self.a_switch = self._config["a_switch"]
        self.lows_particles = []
        self.mids_particles = []
        self.highs_particles = []
        self._multiplier = 1.6 - self._config["blur"] / 17

    def do_once(self):
        super().do_once()
        self.maximum_velocity = (self.r_height + self.r_width / 2) / 10
        # defer things that can't be done when pixel_count is not known
        # this is probably important for most 2d matrix where you want
        # things to be initialized to led length and implied dimensions
        #
        # self.r_width and self.r_height should be used for the (r)ender space
        # as the self.matrix will not exist yet
        #
        # note that self.t_width and self.t_height are the physical dimensions
        #
        # this function will be called once on the first entry to render call
        # in base class twod AND every time there is a config_updated thereafter

    def audio_data_updated(self, data):
        # Grab the filtered melbank
        self.filtered_melbank = self.melbank(
            filtered=False, size=self.pixel_count
        )

        self.lows_idx, self.mids_idx, self.highs_idx = (
            min(
                int(self._multiplier * self.pixel_count * np.mean(i)),
                len(self.filtered_melbank) - 1,
            )
            for i in self.melbank_thirds(filtered=False)
        )
        self.beat_now = data.volume_beat_now()
        self.lows_intensity = np.mean(self.filtered_melbank[self.lows_idx])
        self.mids_intensity = np.mean(self.filtered_melbank[self.mids_idx])
        self.highs_intensity = np.mean(self.filtered_melbank[self.highs_idx])

        for particles, intensity in zip(
            [self.lows_particles, self.mids_particles, self.highs_particles],
            [self.lows_intensity, self.mids_intensity, self.highs_intensity],
        ):
            for particle in particles:
                particle.intensity = intensity

        # Add more particles if there's a beat
        if self.beat_now:
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
            ]  # Red, green, blue
            for particles, color, intensity in zip(
                [
                    self.lows_particles,
                    self.mids_particles,
                    self.highs_particles,
                ],
                colors,
                [
                    self.lows_intensity,
                    self.mids_intensity,
                    self.highs_intensity,
                ],
            ):
                position = np.random.rand(2) * [self.r_width, self.r_height]
                velocity = (
                    np.random.rand(2) - 0.5
                ) * 10  # Random velocity between -5 and 5

                # Ensure velocity does not exceed maximum_velocity
                if np.linalg.norm(velocity) > self.maximum_velocity:
                    velocity = (
                        velocity
                        / np.linalg.norm(velocity)
                        * self.maximum_velocity
                    )

                particles.append(
                    Particle(position, velocity, color, intensity)
                )

    def draw(self):
        # Fade colors for trail effect
        # Update and draw particles
        for particles in [
            self.lows_particles,
            self.mids_particles,
            self.highs_particles,
        ]:
            for (
                particle
            ) in (
                particles.copy()
            ):  # Use copy to avoid modifying list during iteration
                # Update particle position and velocity
                particle.position += particle.velocity
                particle.velocity += (
                    np.random.rand(2) - 0.5
                ) * 2  # Random change between -1 and 1

                # Check if particle is within matrix dimensions
                if (
                    0 <= particle.position[0] < self.matrix.width
                    and 0 <= particle.position[1] < self.matrix.height
                ):
                    # Draw particle
                    r, g, b = particle.color
                    r = int(r * particle.intensity)
                    g = int(g * particle.intensity)
                    b = int(b * particle.intensity)
                    self.matrix.putpixel(
                        tuple(particle.position.astype(int)), (r, g, b)
                    )
                else:
                    # Remove particle if it's outside the matrix
                    particles.remove(particle)
