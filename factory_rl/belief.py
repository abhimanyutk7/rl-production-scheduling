"""Bootstrap condition inference. This module never receives simulator truth."""
import math
import numpy as np


class HealthBelief:
    def __init__(self, config, rng):
        self.config, self.rng = config, rng
        self.health = rng.uniform(config.initial_health_low, 1.0, config.particles)
        sigma = config.wear_rate_sigma
        self.rate = rng.lognormal(-0.5 * sigma ** 2, sigma, config.particles)
        self.weights = np.full(config.particles, 1.0 / config.particles)
        self.fallbacks = 0

    def _condition(self, log_likelihood):
        logs = np.log(np.maximum(self.weights, 1e-300)) + log_likelihood
        peak = np.max(logs)
        if not np.isfinite(peak):
            self.fallbacks += 1
            return
        weights = np.exp(logs - peak)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            self.fallbacks += 1
            return
        self.weights = weights / total
        if 1 / np.sum(self.weights ** 2) < len(weights) / 2:
            positions = (self.rng.random() + np.arange(len(weights))) / len(weights)
            indices = np.searchsorted(np.cumsum(self.weights), positions, side='right')
            indices = np.minimum(indices, len(weights) - 1)
            self.health, self.rate = self.health[indices].copy(), self.rate[indices].copy()
            self.weights.fill(1 / len(weights))

    def predict(self, decrement, failure=None, base=0.0, slope=0.0):
        if decrement:
            cv = self.config.wear_noise_cv
            shock = self.rng.gamma(1 / cv ** 2, cv ** 2, len(self.health)) if cv else 1.0
            self.health = np.maximum(0.0, self.health - decrement * self.rate * shock)
        if failure is not None:
            probability = np.clip(base + slope * (1 - self.health), 0, 1)
            likelihood = probability if failure else 1 - probability
            self._condition(np.log(np.maximum(likelihood, 1e-300)))

    def observe(self, value, noise):
        if not 0 <= value <= 1 or noise < 0 or np.isnan(noise):
            raise ValueError('Invalid condition measurement')
        if math.isinf(noise):
            return
        if noise == 0:
            # An exact measurement identifies health, not the latent wear rate.
            self._condition(-0.5 * ((self.health - value) / 0.005) ** 2)
            self.health.fill(value)
            return
        if value == 0:
            likelihood = np.fromiter((0.5 * math.erfc(h / noise / math.sqrt(2))
                                      for h in self.health), dtype=float)
            logs = np.log(np.maximum(likelihood, 1e-300))
        elif value == 1:
            likelihood = np.fromiter((0.5 * math.erfc((1 - h) / noise / math.sqrt(2))
                                      for h in self.health), dtype=float)
            logs = np.log(np.maximum(likelihood, 1e-300))
        else:
            logs = -0.5 * ((self.health - value) / noise) ** 2
        self._condition(logs)

    def restore(self, health):
        self.health.fill(health)

    def summary(self):
        mean = float(self.weights @ self.health)
        variance = float(self.weights @ ((self.health - mean) ** 2))
        order = np.argsort(self.health)
        quantiles = np.interp([0.05, 0.5, 0.95], np.cumsum(self.weights[order]), self.health[order])
        return dict(mean=mean, std=float(np.sqrt(max(0, variance))),
                    q05=float(quantiles[0]), median=float(quantiles[1]), q95=float(quantiles[2]))
