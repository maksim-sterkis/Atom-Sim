/* High performance rejection sampling worker */
    const FACTORIALS = [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600];
    function factorial(n) {
      if (n < 0) return 1;
      if (n < FACTORIALS.length) return FACTORIALS[n];
      let res = FACTORIALS[FACTORIALS.length - 1];
      for (let i = FACTORIALS.length; i <= n; i++) res *= i;
      return res;
    }

    function assocLaguerre(k, alpha, x) {
      if (k === 0) return 1.0;
      if (k === 1) return 1.0 + alpha - x;
      let prev2 = 1.0;
      let prev1 = 1.0 + alpha - x;
      let current = 0.0;
      for (let j = 1; j < k; j++) {
        current = ((2.0 * j + 1.0 + alpha - x) * prev1 - (j + alpha) * prev2) / (j + 1.0);
        prev2 = prev1;
        prev1 = current;
      }
      return current;
    }

    function assocLegendre(l, m, x) {
      x = Math.max(-1.0, Math.min(1.0, x));
      let pmm = 1.0;
      if (m > 0) {
        const somx2 = Math.sqrt((1.0 - x) * (1.0 + x));
        let fact = 1.0;
        for (let i = 1; i <= m; i++) {
          pmm *= -fact * somx2;
          fact += 2.0;
        }
      }
      if (l === m) return pmm;

      let pmmp1 = x * (2.0 * m + 1.0) * pmm;
      if (l === m + 1) return pmmp1;

      let pll = 0.0;
      let prev2 = pmm;
      let prev1 = pmmp1;
      for (let ll = m + 2; ll <= l; ll++) {
        pll = ((2.0 * ll - 1.0) * x * prev1 - (ll + m - 1.0) * prev2) / (ll - m);
        prev2 = prev1;
        prev1 = pll;
      }
      return pll;
    }

    function radialWavefunction(n, l, Z, r) {
      if (r < 0) return 0.0;
      const rho = (2.0 * Z * r) / n;
      const k = n - l - 1;
      const alpha = 2 * l + 1;
      const normFactor = Math.sqrt(
        Math.pow((2.0 * Z) / n, 3) * (factorial(k) / (2.0 * n * factorial(n + l)))
      );
      return normFactor * Math.exp(-rho / 2.0) * Math.pow(rho, l) * assocLaguerre(k, alpha, rho);
    }

    function realSphericalHarmonic(l, m, theta, phi) {
      const mAbs = Math.abs(m);
      const cosTheta = Math.cos(theta);
      const norm = Math.sqrt(
        ((2.0 * l + 1.0) / (4.0 * Math.PI)) * (factorial(l - mAbs) / factorial(l + mAbs))
      );
      const P = assocLegendre(l, mAbs, cosTheta);
      if (m === 0) {
        return norm * P;
      } else if (m > 0) {
        return Math.SQRT2 * Math.pow(-1, mAbs) * norm * P * Math.cos(m * phi);
      } else {
        return Math.SQRT2 * Math.pow(-1, mAbs) * norm * P * Math.sin(mAbs * phi);
      }
    }

    self.onmessage = function(e) {
      const { jobId, n, l, m, Z, count } = e.data;

      const meanRadius = ((3 * n * n - l * (l + 1)) / (2 * Z));
      const rCut = Math.max(16.0, meanRadius * 2.8 + 6.0);

      // Estimate P_R,max
      let pRMax = 1e-7;
      const steps = 250;
      for (let i = 0; i <= steps; i++) {
        const rTest = (i / steps) * rCut;
        const R = radialWavefunction(n, l, Z, rTest);
        const pR = rTest * rTest * R * R;
        if (pR > pRMax) pRMax = pR;
      }
      pRMax *= 1.05;

      // Estimate angular max
      let yMaxSq = 1e-6;
      for (let i = 0; i <= 60; i++) {
        const cosTheta = -1.0 + (2.0 * i) / 60;
        const theta = Math.acos(cosTheta);
        for (let j = 0; j <= 60; j++) {
          const phi = (j / 60) * Math.PI * 2;
          const Y = realSphericalHarmonic(l, m, theta, phi);
          const ySq = Y * Y;
          if (ySq > yMaxSq) yMaxSq = ySq;
        }
      }
      yMaxSq *= 1.05;

      const positions = new Float32Array(count * 3);
      const phases = new Float32Array(count);
      const radii = new Float32Array(count);
      const densities = new Float32Array(count);
      let maxDensity = 0;

      let sampled = 0;
      while (sampled < count) {
        let r = 0;
        while (true) {
          const rCandidate = Math.random() * rCut;
          const u = Math.random() * pRMax;
          const R = radialWavefunction(n, l, Z, rCandidate);
          const pR = rCandidate * rCandidate * R * R;
          if (u <= pR) {
            r = rCandidate;
            break;
          }
        }

        let theta = 0;
        let phi = 0;
        let psiVal = 0;
        while (true) {
          const cosTheta = -1.0 + Math.random() * 2.0;
          const phiCandidate = Math.random() * Math.PI * 2.0;
          const thetaCandidate = Math.acos(cosTheta);
          const u = Math.random() * yMaxSq;
          const Y = realSphericalHarmonic(l, m, thetaCandidate, phiCandidate);
          if (u <= Y * Y) {
            theta = thetaCandidate;
            phi = phiCandidate;
            const R = radialWavefunction(n, l, Z, r);
            psiVal = R * Y;
            break;
          }
        }

        const sinTheta = Math.sin(theta);
        const idx3 = sampled * 3;
        positions[idx3]     = r * sinTheta * Math.cos(phi);
        positions[idx3 + 1] = r * Math.cos(theta);
        positions[idx3 + 2] = r * sinTheta * Math.sin(phi);

        phases[sampled] = (psiVal >= 0) ? 1.0 : -1.0;
        radii[sampled]  = r;
        densities[sampled] = Math.abs(psiVal);
        if (densities[sampled] > maxDensity) maxDensity = densities[sampled];
        sampled++;
      }

      for (let i = 0; i < count; i++) densities[i] = maxDensity > 0 ? densities[i] / maxDensity : 0;
      self.postMessage(
        { jobId, positions, phases, radii, densities, count },
        [positions.buffer, phases.buffer, radii.buffer, densities.buffer]
      );
    };