// PINN Harmonic Oscillator Frontend
class PINNApp {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.currentData = null;
        this.currentTrainingResult = null;
        this.currentTimeRange = 10;

        this.initCharts();
        this.initControls();
        this.setupEventListeners();
        this.updateAnalyticalSolution();
    }

    initCharts() {
        // Prediction chart
        this.predictionChart = new Chart(
            document.getElementById('prediction-chart'),
            {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'PINN Prediction',
                            data: [],
                            borderColor: '#3498db',
                            backgroundColor: 'rgba(52, 152, 219, 0.1)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4,
                            fill: true
                        },
                        {
                            label: 'Analytical Solution',
                            data: [],
                            borderColor: '#e74c3c',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            tension: 0.4,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 500 },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Time (t)', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Position (x)', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#aaa' }
                        }
                    }
                }
            }
        );

        // Training history chart
        this.trainingChart = new Chart(
            document.getElementById('training-chart'),
            {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Total Loss',
                            data: [],
                            borderColor: '#667eea',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4,
                            fill: false
                        },
                        {
                            label: 'IC Loss',
                            data: [],
                            borderColor: '#2ecc71',
                            borderWidth: 1,
                            pointRadius: 0,
                            tension: 0.3,
                            fill: false
                        },
                        {
                            label: 'Physics Loss',
                            data: [],
                            borderColor: '#f39c12',
                            borderWidth: 1,
                            pointRadius: 0,
                            tension: 0.3,
                            fill: false
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 500 },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Epoch', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            type: 'logarithmic',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Loss', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#aaa' }
                        }
                    }
                }
            }
        );

        // Residual chart
        this.residualChart = new Chart(
            document.getElementById('residual-chart'),
            {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Residual',
                            data: [],
                            borderColor: '#9b59b6',
                            backgroundColor: 'rgba(155, 89, 182, 0.2)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.4,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 500 },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Time (t)', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Residual', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#aaa' }
                        }
                    }
                }
            }
        );

        // Phase space chart
        this.phaseSpaceChart = new Chart(
            document.getElementById('phase-space-chart'),
            {
                type: 'line',
                data: {
                    datasets: [
                        {
                            label: 'Phase Space (PINN)',
                            data: [],
                            borderColor: '#1abc9c',
                            backgroundColor: 'rgba(26, 188, 156, 0.2)',
                            borderWidth: 2,
                            pointRadius: 0,
                            tension: 0.3,
                            fill: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 500 },
                    parsing: false,
                    scales: {
                        x: {
                            type: 'linear',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Position (x)', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        },
                        y: {
                            type: 'linear',
                            grid: { color: 'rgba(255,255,255,0.1)' },
                            title: { text: 'Velocity (dx/dt)', display: true, color: '#aaa' },
                            ticks: { color: '#888' }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: { color: '#aaa' }
                        }
                    }
                }
            }
        );
    }

    initControls() {
        // Initialize sliders with value displays and numeric inputs
        const controls = [
            { id: 'omega', display: 'omega-value', input: 'omega-input' },
            { id: 'damping_coefficient', display: 'damping_coefficient-value', input: 'damping_coefficient-input' },
            { id: 't_max', display: 't_max-value', input: 't_max-input' },
            { id: 'epochs', display: 'epochs-value', input: 'epochs-input' },
            { id: 'hidden_dim', display: 'hidden_dim-value', input: 'hidden_dim-input' },
            { id: 'learning_rate', display: 'learning_rate-value', input: 'learning_rate-input' },
            { id: 'weight_decay', display: 'weight_decay-value', input: 'weight_decay-input' },
            { id: 'lambda_ic', display: 'lambda_ic-value', input: 'lambda_ic-input' },
            { id: 'lambda_physics', display: 'lambda_physics-value', input: 'lambda_physics-input' },
            { id: 'num_domain_points', display: 'num_domain_points-value', input: 'num_domain_points-input' }
        ];

        controls.forEach(control => {
            const slider = document.getElementById(control.id);
            const display = document.getElementById(control.display);
            const numericInput = document.getElementById(control.input);

            // Initialize values
            display.textContent = slider.value;
            numericInput.value = slider.value;

            // Slider change handler
            slider.addEventListener('input', () => {
                const value = slider.value;
                display.textContent = value;
                numericInput.value = value;

                // Update analytical solution in real-time for relevant parameters
                if (control.id === 'omega' || control.id === 't_max' || control.id === 'damping_coefficient') {
                    this.updateAnalyticalSolution();
                }
            });

            // Numeric input change handler
            numericInput.addEventListener('input', () => {
                const value = numericInput.value;
                // Validate range
                const min = parseFloat(slider.min);
                const max = parseFloat(slider.max);
                const step = parseFloat(slider.step);

                let validatedValue = Math.max(min, Math.min(max, parseFloat(value) || min));

                // Round to nearest step
                validatedValue = Math.round(validatedValue / step) * step;

                slider.value = validatedValue;
                display.textContent = validatedValue;
                numericInput.value = validatedValue;

                // Update analytical solution in real-time for relevant parameters
                if (control.id === 'omega' || control.id === 't_max' || control.id === 'damping_coefficient') {
                    this.updateAnalyticalSolution();
                }
            });
        });
    }

    computeAnalyticalSolution(omega, damping_coefficient, t_max, numPoints = 500) {
        // Analytical solution for damped oscillator
        // x(t) = A * exp(-γt/2) * cos(ω_d * t + φ)
        // where γ = damping_coefficient, ω_d = sqrt(ω^2 - γ^2/4)
        // For undamped: γ = 0, so ω_d = ω, and x(t) = cos(ω * t)

        const t_values = [];
        const x_values = [];
        const dt = t_max / (numPoints - 1);

        // For undamped case
        if (damping_coefficient === 0) {
            for (let i = 0; i < numPoints; i++) {
                const t = i * dt;
                t_values.push(t);
                x_values.push(Math.cos(omega * t));
            }
        } else {
            // For damped case, we need to compute the damped frequency
            const gamma = damping_coefficient;
            const omega_squared = omega * omega;
            const gamma_squared_over_4 = gamma * gamma / 4;

            // Check if overdamped or underdamped
            if (gamma_squared_over_4 >= omega_squared) {
                // Overdamped case
                const r1 = -gamma/2 + Math.sqrt(gamma_squared_over_4 - omega_squared);
                const r2 = -gamma/2 - Math.sqrt(gamma_squared_over_4 - omega_squared);
                // x(t) = C1 * exp(r1*t) + C2 * exp(r2*t)
                // With initial conditions x(0) = 1, v(0) = 0:
                // C1 + C2 = 1, C1*r1 + C2*r2 = 0
                const C1 = -r2 / (r1 - r2);
                const C2 = r1 / (r1 - r2);
                for (let i = 0; i < numPoints; i++) {
                    const t = i * dt;
                    t_values.push(t);
                    x_values.push(C1 * Math.exp(r1 * t) + C2 * Math.exp(r2 * t));
                }
            } else {
                // Underdamped case (most common)
                const omega_d = Math.sqrt(omega_squared - gamma_squared_over_4);
                // x(t) = exp(-γt/2) * (cos(ω_d * t) + (γ/(2*ω_d)) * sin(ω_d * t))
                // But with initial conditions x(0) = 1, v(0) = 0:
                // x(0) = 1 = A * cos(φ) => A = 1, φ = 0
                // v(0) = 0 = -A * γ/2 * cos(φ) + A * ω_d * sin(φ)
                // 0 = -γ/2 + ω_d * tan(φ) => tan(φ) = γ/(2*ω_d) => φ = atan(γ/(2*ω_d))
                // Actually, for simpler case, using just cos(ω_d * t) with amplitude decay
                for (let i = 0; i < numPoints; i++) {
                    const t = i * dt;
                    t_values.push(t);
                    x_values.push(Math.exp(-gamma * t / 2) * Math.cos(omega_d * t));
                }
            }
        }
        return { t_values, x_values };
    }

    updateAnalyticalSolution() {
        const omega = parseFloat(document.getElementById('omega').value);
        const damping_coefficient = parseFloat(document.getElementById('damping_coefficient').value);
        const t_max = parseFloat(document.getElementById('t_max').value);

        const { t_values, x_values } = this.computeAnalyticalSolution(omega, damping_coefficient, t_max);

        // Update prediction chart analytical solution dataset
        this.predictionChart.data.datasets[1].data = t_values.map((t, i) => ({ x: t, y: x_values[i] }));
        this.predictionChart.update();
    }

    setupEventListeners() {
        document.getElementById('train-btn').addEventListener('click', () => this.trainModel());
        document.getElementById('reset-btn').addEventListener('click', () => this.resetToDefaults());

        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                // Update active tab
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                // Update active content
                const tab = btn.dataset.tab;
                document.querySelectorAll('.tab-content').forEach(c => {
                    c.classList.remove('active');
                    if (c.id === tab + '-tab') {
                        c.classList.add('active');
                    }
                });
                
                // Refresh chart if data exists
                if (this.currentTrainingResult) {
                    if (tab === 'training') {
                        this.updateTrainingChart();
                    } else if (tab === 'prediction') {
                        this.updateAllCharts();
                    } else if (tab === 'residual') {
                        this.residualChart.update();
                    } else if (tab === 'phase-space') {
                        this.phaseSpaceChart.update();
                    }
                }
            });
        });
        
        // Enter key on any input triggers training
        document.querySelectorAll('input').forEach(input => {
            if (input.type === 'range') return;
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.trainModel();
            });
        });
    }

    async trainModel() {
        const trainBtn = document.getElementById('train-btn');
        const statusDiv = document.getElementById('status');
        const progressContainer = document.getElementById('progress-container');
        const progress = document.getElementById('progress');
        const progressText = document.getElementById('progress-text');

        // Disable train button
        trainBtn.disabled = true;
        statusDiv.className = 'status-message info';
        statusDiv.textContent = 'Training in progress...';
        progressContainer.style.display = 'block';
        progress.style.width = '0%';

        try {
            const data = {
                omega: parseFloat(document.getElementById('omega').value),
                damping_coefficient: parseFloat(document.getElementById('damping_coefficient').value),
                t_max: parseFloat(document.getElementById('t_max').value),
                num_domain_points: parseInt(document.getElementById('num_domain_points').value),
                hidden_dim: parseInt(document.getElementById('hidden_dim').value),
                epochs: parseInt(document.getElementById('epochs').value),
                learning_rate: parseFloat(document.getElementById('learning_rate').value),
                weight_decay: parseFloat(document.getElementById('weight_decay').value),
                lambda_ic: parseFloat(document.getElementById('lambda_ic').value),
                lambda_physics: parseFloat(document.getElementById('lambda_physics').value)
            };

            // Simulate progress updates
            const progressInterval = setInterval(() => {
                progress.style.width = Math.min(
                    (Date.now() % 10000) / 100 + '%',
                    '95%'
                );
            }, 100);

            // Make API call
            const response = await fetch(`${this.serverUrl}/api/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            clearInterval(progressInterval);
            progress.style.width = '100%';

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Training failed');
            }

            this.currentTrainingResult = await response.json();

            // Update UI
            this.updateAllCharts();
            statusDiv.className = 'status-message success';
            statusDiv.innerHTML = `Training completed! Final loss: ${this.currentTrainingResult.final_loss.toFixed(6)}`;

        } catch (error) {
            console.error('Training error:', error);
            statusDiv.className = 'status-message error';
            statusDiv.textContent = `Error: ${error.message}`;
        } finally {
            trainBtn.disabled = false;
            progressContainer.style.display = 'none';
        }
    }

    updateAllCharts() {
        if (!this.currentTrainingResult) return;

        const result = this.currentTrainingResult;

        // Debug: Check data values
        console.log('t values sample:', result.t.slice(0, 5));
        console.log('x_pred values sample:', result.x_pred.slice(0, 5));

        // Update prediction chart - use array format [x, y] with parsing: false
        this.predictionChart.data.datasets[0].data = result.t.map((t, i) => ({ x: t, y: result.x_pred[i] }));
        this.predictionChart.data.datasets[1].data = result.t.map((t, i) => ({ x: t, y: result.x_analytical[i] }));

        this.predictionChart.update();

        // Update training chart
        this.updateTrainingChart();

        // Update residual chart with proper scaling - use object format {x, y}
        this.residualChart.data.datasets[0].data = result.t.map((t, i) => ({ x: t, y: result.residual[i] }));
        this.residualChart.update();

        // Update phase space chart - use object format {x, y}
        this.phaseSpaceChart.data.datasets[0].data = result.t.map((t, i) => ({ x: result.x_pred[i], y: result.velocity[i] }));
        this.phaseSpaceChart.update();
    }

    updateTrainingChart() {
        if (!this.currentTrainingResult) return;

        const history = this.currentTrainingResult.history;

        // Use array format [epoch, loss] with parsing: false
        this.trainingChart.data.datasets[0].data = history.train_loss.map((loss, i) => ({ x: i, y: loss }));
        this.trainingChart.data.datasets[1].data = history.ic_loss.map((loss, i) => ({ x: i, y: loss }));
        this.trainingChart.data.datasets[2].data = history.physics_loss.map((loss, i) => ({ x: i, y: loss }));

        document.getElementById('final-loss').textContent = history.train_loss[history.train_loss.length - 1].toFixed(6);
        this.trainingChart.update();
    }

    resetToDefaults() {
        // Default parameter values
        const defaults = {
            omega: 1.0,
            t_max: 10,
            epochs: 1000,
            hidden_dim: 50,
            learning_rate: 0.02,
            weight_decay: 0.01,
            lambda_ic: 10,
            lambda_physics: 1.0,
            num_domain_points: 500
        };

        // Reset sliders
        document.getElementById('omega').value = defaults.omega;
        document.getElementById('t_max').value = defaults.t_max;
        document.getElementById('epochs').value = defaults.epochs;
        document.getElementById('hidden_dim').value = defaults.hidden_dim;
        document.getElementById('learning_rate').value = defaults.learning_rate;
        document.getElementById('weight_decay').value = defaults.weight_decay;
        document.getElementById('lambda_ic').value = defaults.lambda_ic;
        document.getElementById('lambda_physics').value = defaults.lambda_physics;
        document.getElementById('num_domain_points').value = defaults.num_domain_points;

        // Reset value displays
        document.getElementById('omega-value').textContent = defaults.omega;
        document.getElementById('t_max-value').textContent = defaults.t_max;
        document.getElementById('epochs-value').textContent = defaults.epochs;
        document.getElementById('hidden_dim-value').textContent = defaults.hidden_dim;
        document.getElementById('learning_rate-value').textContent = defaults.learning_rate;
        document.getElementById('weight_decay-value').textContent = defaults.weight_decay;
        document.getElementById('lambda_ic-value').textContent = defaults.lambda_ic;
        document.getElementById('lambda_physics-value').textContent = defaults.lambda_physics;
        document.getElementById('num_domain_points-value').textContent = defaults.num_domain_points;

        // Clear results
        this.currentTrainingResult = null;
        this.predictionChart.data.datasets[0].data = [];
        this.predictionChart.update();

        // Update analytical solution with defaults
        this.updateAnalyticalSolution();
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.pinnApp = new PINNApp();
});