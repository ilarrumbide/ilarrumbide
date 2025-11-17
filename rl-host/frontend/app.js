// Restaurant RL Host - Frontend Application
// Retro game-style visualization with Canvas and WebSocket

class RestaurantApp {
    constructor() {
        this.canvas = document.getElementById('restaurantCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.ws = null;
        this.state = null;
        this.currentTime = 0;
        this.simulationRunning = false;

        // Colors (retro palette)
        this.colors = {
            background: '#0a0e27',
            grid: '#1a1f3a',
            tableAvailable: '#00ff9f',
            tableOccupied: '#ff3366',
            tableCleaning: '#ffaa00',
            customer: '#00d4ff',
            text: '#e0e0e0',
            zone: {
                inside: 'rgba(0, 255, 159, 0.1)',
                outside: 'rgba(0, 212, 255, 0.1)',
                window: 'rgba(255, 170, 0, 0.1)',
                bar: 'rgba(255, 51, 102, 0.1)'
            }
        };

        this.init();
    }

    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.startRenderLoop();
        this.loadInitialState();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            this.log('Connected to server', 'success');
            this.showConnectionStatus(true);
        };

        this.ws.onclose = () => {
            this.log('Disconnected from server', 'error');
            this.showConnectionStatus(false);
            setTimeout(() => this.setupWebSocket(), 3000);
        };

        this.ws.onerror = (error) => {
            this.log('WebSocket error', 'error');
            console.error(error);
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
        };
    }

    handleWebSocketMessage(message) {
        switch (message.type) {
            case 'initial_state':
            case 'state_update':
                this.state = message.state;
                this.currentTime = message.current_time || 0;
                this.updateUI();
                if (message.last_action) {
                    this.log(`AI: ${message.last_action}`, 'info');
                }
                break;

            case 'customer_arrived':
                this.log(`Customer arrived: Group of ${message.group.size}`, 'info');
                this.requestState();
                break;

            case 'customer_seated':
                this.log(`Customer seated at table(s) ${message.table_ids.join(', ')}`, 'success');
                this.requestState();
                break;

            case 'customer_left':
                this.log(`Customer left (impatient)`, 'warning');
                this.requestState();
                break;

            case 'restaurant_reset':
                this.state = message.state;
                this.currentTime = 0;
                this.updateUI();
                this.log('Restaurant reset', 'info');
                break;

            case 'simulation_started':
                this.simulationRunning = true;
                this.log(`Simulation started: ${message.num_customers} customers`, 'success');
                break;

            case 'simulation_stopped':
                this.simulationRunning = false;
                this.log('Simulation stopped', 'info');
                break;

            case 'ai_action_executed':
                this.log(`AI executed: ${message.action} (reward: ${message.reward.toFixed(2)})`, 'info');
                this.requestState();
                break;
        }
    }

    requestState() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type: 'request_state' }));
        }
    }

    async loadInitialState() {
        try {
            const response = await fetch('/api/state');
            const data = await response.json();
            if (data.status === 'success') {
                this.state = data.state;
                this.currentTime = data.current_time;
                this.simulationRunning = data.simulation_running;
                this.updateUI();
            }
        } catch (error) {
            console.error('Failed to load initial state:', error);
        }
    }

    setupEventListeners() {
        // Reset button
        document.getElementById('btnReset').addEventListener('click', () => {
            this.resetRestaurant();
        });

        // Simulation controls
        document.getElementById('btnStartSim').addEventListener('click', () => {
            this.startSimulation();
        });

        document.getElementById('btnStopSim').addEventListener('click', () => {
            this.stopSimulation();
        });

        // AI controls
        document.getElementById('btnAIDecision').addEventListener('click', () => {
            this.getAIDecision();
        });

        document.getElementById('btnExecuteAI').addEventListener('click', () => {
            this.executeAIAction();
        });

        // Manual controls
        document.getElementById('btnAddCustomer').addEventListener('click', () => {
            this.showAddCustomerModal();
        });

        document.getElementById('btnCancelAdd').addEventListener('click', () => {
            this.hideAddCustomerModal();
        });

        document.getElementById('addCustomerForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.addCustomer();
        });

        // Canvas click for table selection
        this.canvas.addEventListener('click', (e) => {
            this.handleCanvasClick(e);
        });
    }

    startRenderLoop() {
        const render = () => {
            this.render();
            requestAnimationFrame(render);
        };
        render();
    }

    render() {
        if (!this.state) return;

        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;

        // Clear canvas
        ctx.fillStyle = this.colors.background;
        ctx.fillRect(0, 0, width, height);

        // Draw grid
        this.drawGrid(ctx, width, height);

        // Draw zones
        this.drawZones(ctx, width, height);

        // Draw tables
        this.drawTables(ctx, width, height);

        // Draw legend
        this.drawLegend(ctx, width, height);
    }

    drawGrid(ctx, width, height) {
        ctx.strokeStyle = this.colors.grid;
        ctx.lineWidth = 1;

        const gridSize = 80;

        // Vertical lines
        for (let x = 0; x <= width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
    }

    drawZones(ctx, width, height) {
        // Zone labels
        ctx.font = 'bold 16px VT323';
        ctx.fillStyle = this.colors.text;

        // Inside zone (top)
        ctx.fillStyle = this.colors.zone.inside;
        ctx.fillRect(80, 80, 320, 120);
        ctx.fillStyle = this.colors.text;
        ctx.fillText('INSIDE', 200, 140);

        // Outside zone (bottom)
        ctx.fillStyle = this.colors.zone.outside;
        ctx.fillRect(80, 240, 320, 120);
        ctx.fillStyle = this.colors.text;
        ctx.fillText('OUTSIDE', 200, 300);

        // Window zone
        ctx.fillStyle = this.colors.zone.window;
        ctx.fillRect(80, 80, 160, 120);
        ctx.fillStyle = this.colors.text;
        ctx.fillText('WINDOW', 110, 100);

        // Bar zone
        ctx.fillStyle = this.colors.zone.bar;
        ctx.fillRect(320, 80, 80, 120);
        ctx.fillStyle = this.colors.text;
        ctx.fillText('BAR', 330, 100);
    }

    drawTables(ctx, width, height) {
        if (!this.state || !this.state.tables) return;

        const tableWidth = 60;
        const tableHeight = 50;
        const baseX = 100;
        const baseY = 120;
        const spacingX = 80;
        const spacingY = 120;

        this.state.tables.forEach(table => {
            const x = baseX + table.coordinates[0] * spacingX;
            const y = baseY + table.coordinates[1] * spacingY;

            // Determine color based on status
            let color = this.colors.tableAvailable;
            if (table.status === 'occupied') {
                color = this.colors.tableOccupied;
            } else if (table.status === 'cleaning') {
                color = this.colors.tableCleaning;
            }

            // Draw table
            ctx.fillStyle = color;
            ctx.fillRect(x - tableWidth / 2, y - tableHeight / 2, tableWidth, tableHeight);

            // Draw border
            ctx.strokeStyle = this.colors.text;
            ctx.lineWidth = 2;
            ctx.strokeRect(x - tableWidth / 2, y - tableHeight / 2, tableWidth, tableHeight);

            // Draw table number and capacity
            ctx.fillStyle = this.colors.background;
            ctx.font = 'bold 14px VT323';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(`T${table.id}`, x, y - 10);
            ctx.fillText(`[${table.capacity}]`, x, y + 10);

            // Glow effect for occupied tables
            if (table.status === 'occupied') {
                ctx.shadowBlur = 15;
                ctx.shadowColor = color;
                ctx.strokeRect(x - tableWidth / 2, y - tableHeight / 2, tableWidth, tableHeight);
                ctx.shadowBlur = 0;
            }
        });

        ctx.textAlign = 'left';
        ctx.textBaseline = 'alphabetic';
    }

    drawLegend(ctx, width, height) {
        const legendX = width - 200;
        const legendY = 20;

        ctx.font = '14px VT323';
        ctx.fillStyle = this.colors.text;
        ctx.fillText('LEGEND:', legendX, legendY);

        const items = [
            { color: this.colors.tableAvailable, text: 'Available' },
            { color: this.colors.tableOccupied, text: 'Occupied' },
            { color: this.colors.tableCleaning, text: 'Cleaning' }
        ];

        items.forEach((item, index) => {
            const y = legendY + 25 + index * 25;

            ctx.fillStyle = item.color;
            ctx.fillRect(legendX, y - 12, 20, 15);

            ctx.strokeStyle = this.colors.text;
            ctx.strokeRect(legendX, y - 12, 20, 15);

            ctx.fillStyle = this.colors.text;
            ctx.fillText(item.text, legendX + 30, y);
        });
    }

    updateUI() {
        if (!this.state) return;

        // Update statistics
        const stats = this.state.statistics || {};
        document.getElementById('statTime').textContent = this.formatTime(this.currentTime);
        document.getElementById('statQueue').textContent = this.state.waiting_queue?.length || 0;
        document.getElementById('statSeated').textContent = this.state.seated_groups?.length || 0;
        document.getElementById('statServed').textContent = stats.total_served || 0;
        document.getElementById('statLost').textContent = stats.total_lost || 0;

        // Calculate satisfaction rate
        const served = stats.total_served || 0;
        const lost = stats.total_lost || 0;
        const satisfaction = served + lost > 0 ? (served / (served + lost) * 100).toFixed(1) : 100;
        document.getElementById('statSatisfaction').textContent = `${satisfaction}%`;

        // Calculate available tables
        const available = this.state.tables?.filter(t => t.status === 'available').length || 0;
        document.getElementById('statAvailable').textContent = available;

        // Calculate efficiency
        const totalTables = this.state.tables?.length || 10;
        const occupancy = totalTables - available;
        const efficiency = ((occupancy / totalTables) * 100).toFixed(0);
        document.getElementById('statEfficiency').textContent = `${efficiency}%`;

        // Update queue
        this.updateQueue();
    }

    updateQueue() {
        const queueList = document.getElementById('queueList');
        const queue = this.state.waiting_queue || [];

        if (queue.length === 0) {
            queueList.innerHTML = '<p class="empty-message">No customers waiting</p>';
            return;
        }

        queueList.innerHTML = queue.map(group => {
            const waitTime = this.currentTime - group.arrival_time;
            const mood = this.getMoodEmoji(group.mood);

            return `
                <div class="queue-item" data-group-id="${group.id}">
                    <div class="queue-item-header">
                        <span class="queue-item-mood">${mood}</span>
                        <span class="queue-item-size">Group of ${group.size}</span>
                    </div>
                    <div class="queue-item-zone">Prefers: ${group.zone_preference}</div>
                    <div class="queue-item-wait">Waiting: ${waitTime.toFixed(0)} min | Patience: ${group.patience_minutes.toFixed(0)} min</div>
                </div>
            `;
        }).join('');
    }

    getMoodEmoji(mood) {
        const moodMap = {
            'happy': '😊',
            'content': '🙂',
            'neutral': '😐',
            'impatient': '😟',
            'angry': '😠',
            'left': '🚶'
        };
        return moodMap[mood] || '😐';
    }

    formatTime(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = Math.floor(minutes % 60);
        return `${hours}:${mins.toString().padStart(2, '0')}`;
    }

    log(message, type = 'info') {
        const logContainer = document.getElementById('eventLog');
        const entry = document.createElement('p');
        entry.className = `log-entry ${type}`;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;

        logContainer.insertBefore(entry, logContainer.firstChild);

        // Keep only last 50 entries
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }

    showConnectionStatus(connected) {
        let statusEl = document.querySelector('.connection-status');

        if (!statusEl) {
            statusEl = document.createElement('div');
            statusEl.className = 'connection-status';
            document.body.appendChild(statusEl);
        }

        statusEl.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
        statusEl.textContent = connected ? '● CONNECTED' : '● DISCONNECTED';
    }

    async resetRestaurant() {
        try {
            const response = await fetch('/api/reset', { method: 'POST' });
            const data = await response.json();
            if (data.status === 'success') {
                this.log('Restaurant reset successfully', 'success');
            }
        } catch (error) {
            this.log('Failed to reset restaurant', 'error');
        }
    }

    async startSimulation() {
        const scenarioType = document.getElementById('scenarioType').value;
        const speedMultiplier = parseFloat(document.getElementById('speedMultiplier').value);

        try {
            const response = await fetch('/api/simulation/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    duration_minutes: 480,
                    scenario_type: scenarioType,
                    speed_multiplier: speedMultiplier
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                this.simulationRunning = true;
                this.log(`Simulation started with ${data.num_customers} customers`, 'success');
            }
        } catch (error) {
            this.log('Failed to start simulation', 'error');
        }
    }

    async stopSimulation() {
        try {
            const response = await fetch('/api/simulation/stop', { method: 'POST' });
            if (response.ok) {
                this.simulationRunning = false;
                this.log('Simulation stopped', 'info');
            }
        } catch (error) {
            this.log('Failed to stop simulation', 'error');
        }
    }

    async getAIDecision() {
        try {
            const response = await fetch('/api/ai/decision');
            const data = await response.json();

            if (data.status === 'success') {
                const aiPanel = document.getElementById('aiDecision');
                aiPanel.innerHTML = `<strong>AI Recommendation:</strong><br>${data.decision}`;

                if (data.current_group) {
                    aiPanel.innerHTML += `<br><br><em>For group of ${data.current_group.size} (${data.current_group.zone_preference})</em>`;
                }

                this.log(`AI suggests: ${data.decision}`, 'info');
            }
        } catch (error) {
            this.log('Failed to get AI decision', 'error');
        }
    }

    async executeAIAction() {
        try {
            const response = await fetch('/api/ai/execute', { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                this.log(`AI executed: ${data.action} (Reward: ${data.reward.toFixed(2)})`, 'success');
            }
        } catch (error) {
            this.log('Failed to execute AI action', 'error');
        }
    }

    showAddCustomerModal() {
        const modal = document.getElementById('addCustomerModal');
        modal.classList.add('show');
    }

    hideAddCustomerModal() {
        const modal = document.getElementById('addCustomerModal');
        modal.classList.remove('show');
    }

    async addCustomer() {
        const size = parseInt(document.getElementById('inputGroupSize').value);
        const zone = document.getElementById('inputZone').value;
        const patience = parseFloat(document.getElementById('inputPatience').value);
        const duration = parseFloat(document.getElementById('inputDuration').value);

        try {
            const response = await fetch('/api/customer/add', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    size,
                    zone_preference: zone,
                    alternative_zones: [],
                    patience_minutes: patience,
                    expected_dining_minutes: duration,
                    special_requirements: []
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                this.log(`Added customer group (${size} people)`, 'success');
                this.hideAddCustomerModal();
            }
        } catch (error) {
            this.log('Failed to add customer', 'error');
        }
    }

    handleCanvasClick(e) {
        // Get click coordinates relative to canvas
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

        // Check if clicked on a table
        const clickedTable = this.getTableAtPosition(x, y);

        if (clickedTable) {
            this.log(`Clicked table ${clickedTable.id} (${clickedTable.zone}, capacity: ${clickedTable.capacity})`, 'info');
        }
    }

    getTableAtPosition(x, y) {
        if (!this.state || !this.state.tables) return null;

        const baseX = 100;
        const baseY = 120;
        const spacingX = 80;
        const spacingY = 120;
        const tableWidth = 60;
        const tableHeight = 50;

        for (const table of this.state.tables) {
            const tx = baseX + table.coordinates[0] * spacingX;
            const ty = baseY + table.coordinates[1] * spacingY;

            if (x >= tx - tableWidth / 2 && x <= tx + tableWidth / 2 &&
                y >= ty - tableHeight / 2 && y <= ty + tableHeight / 2) {
                return table;
            }
        }

        return null;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.restaurantApp = new RestaurantApp();
});
