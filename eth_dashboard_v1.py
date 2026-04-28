#!/usr/bin/env python3
import http.server
import socketserver
import json
import os
import sqlite3
import glob

PORT = 8080
DB_PATH = "trading_system_v31.db"
STATE_DIR = ".bot_state"
REGIME_FILE = "regime_state.json"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eth-Bot Orchestrator | Command Center</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0a0b10;
            --card-bg: rgba(255, 255, 255, 0.05);
            --border: rgba(255, 255, 255, 0.1);
            --accent: #3b82f6;
            --accent-glow: rgba(59, 130, 246, 0.5);
            --text: #e2e8f0;
            --text-dim: #94a3b8;
            --bull: #10b981;
            --crash: #ef4444;
            --range: #f59e0b;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: var(--bg); 
            color: var(--text); 
            font-family: 'Inter', sans-serif;
            overflow-x: hidden;
            background-image: radial-gradient(circle at 50% -20%, #1e1b4b 0%, var(--bg) 80%);
        }

        .container { max-width: 1400px; margin: 0 auto; padding: 2rem; }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 3rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }

        .logo { font-size: 1.5rem; font-weight: 700; letter-spacing: -1px; }
        .logo span { color: var(--accent); }

        .regime-badge {
            padding: 0.5rem 1.5rem;
            border-radius: 999px;
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.875rem;
            letter-spacing: 1px;
            background: rgba(0,0,0,0.5);
            border: 1px solid var(--border);
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
        }

        .BULL { border-color: var(--bull); color: var(--bull); box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }
        .CRASH { border-color: var(--crash); color: var(--crash); box-shadow: 0 0 15px rgba(239, 68, 68, 0.3); }
        .RANGE { border-color: var(--range); color: var(--range); box-shadow: 0 0 15px rgba(245, 158, 11, 0.3); }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            transition: transform 0.2s;
        }
        .card:hover { transform: translateY(-4px); }

        .card-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
        .card-value { font-size: 2rem; font-weight: 700; }
        .card-sub { font-size: 0.875rem; color: var(--text-dim); margin-top: 0.5rem; }

        .bot-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 3rem;
        }

        .bot-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.25rem;
            position: relative;
            overflow: hidden;
        }
        .bot-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 4px; height: 100%;
            background: var(--accent);
        }

        .bot-name { font-weight: 700; margin-bottom: 1rem; display: flex; justify-content: space-between; }
        .bot-status { font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px; background: rgba(255,255,255,0.1); }
        .bot-pnl { font-size: 1.25rem; font-weight: 600; margin: 0.5rem 0; }
        .pos { color: var(--bull); }
        .neg { color: var(--crash); }

        .reasoning-panel {
            background: rgba(0,0,0,0.3);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        .reasoning-title { font-family: 'Inter', sans-serif; font-weight: 700; margin-bottom: 1rem; color: var(--accent); }

        .trade-log {
            width: 100%;
            border-collapse: collapse;
            margin-top: 2rem;
            font-size: 0.875rem;
        }
        .trade-log th { text-align: left; padding: 1rem; color: var(--text-dim); border-bottom: 1px solid var(--border); }
        .trade-log td { padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); }

        @keyframes pulse {
            0% { opacity: 0.5; }
            50% { opacity: 1; }
            100% { opacity: 0.5; }
        }
        .live-indicator {
            display: inline-block;
            width: 8px; height: 8px;
            background: var(--bull);
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">ETH<span>BOT</span> ORCHESTRATOR</div>
            <div id="regime-display" class="regime-badge">INITIALIZING...</div>
        </header>

        <div class="stats-grid">
            <div class="card">
                <div class="card-label">Network Conviction</div>
                <div id="conviction-value" class="card-value">--%</div>
                <div id="scaling-sub" class="card-sub">Capital Scaling: --</div>
            </div>
            <div class="card">
                <div class="card-label">Active Positions</div>
                <div id="pos-count" class="card-value">0</div>
                <div id="pnl-total" class="card-sub">Total Unrealized: $0.00</div>
            </div>
            <div class="card">
                <div class="card-label">System Health</div>
                <div class="card-value"><span class="live-indicator"></span>LIVE</div>
                <div id="last-update" class="card-sub">Last update: Just now</div>
            </div>
        </div>

        <h2 style="margin-bottom: 1.5rem;">Bot Fleet</h2>
        <div id="bot-grid" class="bot-grid">
            <!-- Bot cards will be injected here -->
        </div>

        <div class="reasoning-panel">
            <div class="reasoning-title">Navigator Reasoning (LLM)</div>
            <div id="navigator-notes">Awaiting analysis from Jules...</div>
        </div>

        <table class="trade-log">
            <thead>
                <tr>
                    <th>Timestamp</th>
                    <th>Bot</th>
                    <th>Side</th>
                    <th>Price</th>
                    <th>PnL</th>
                </tr>
            </thead>
            <tbody id="trade-history">
                <!-- Trades will be injected here -->
            </tbody>
        </table>
    </div>

    <script>
        async function updateDashboard() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();

                // Update Regime
                const rd = document.getElementById('regime-display');
                rd.textContent = data.regime.current_regime || 'UNKNOWN';
                rd.className = 'regime-badge ' + (data.regime.current_regime || 'RANGE');

                // Update Stats
                document.getElementById('conviction-value').textContent = Math.round(data.regime.conviction * 100) + '%';
                document.getElementById('scaling-sub').textContent = 'Capital Scaling: ' + data.regime.conviction.toFixed(2);
                document.getElementById('navigator-notes').textContent = data.regime.advisor_notes || 'No notes.';

                // Update Bot Grid
                const grid = document.getElementById('bot-grid');
                grid.innerHTML = '';
                let totalUnreal = 0;
                let activeCount = 0;

                data.bots.forEach(bot => {
                    const card = document.createElement('div');
                    card.className = 'bot-card';
                    const isPos = bot.position.qty > 0;
                    if(isPos) activeCount++;
                    
                    card.innerHTML = `
                        <div class="bot-name">
                            ${bot.bot_id.split('_')[0].toUpperCase()}
                            <span class="bot-status">${isPos ? 'OPEN' : 'IDLE'}</span>
                        </div>
                        <div class="card-label">Inventory</div>
                        <div style="font-size: 1.25rem; font-weight: 700;">${bot.position.qty.toFixed(3)} ETH</div>
                        <div class="bot-pnl ${isPos ? 'pos' : ''}">
                            Avg: $${Math.round(bot.position.avg_entry)}
                        </div>
                    `;
                    grid.appendChild(card);
                });

                document.getElementById('pos-count').textContent = activeCount;
                
                // Update History
                const history = document.getElementById('trade-history');
                history.innerHTML = '';
                data.history.forEach(t => {
                    const row = document.createElement('tr');
                    const pnlClass = t.pnl > 0 ? 'pos' : (t.pnl < 0 ? 'neg' : '');
                    row.innerHTML = `
                        <td style="color: var(--text-dim)">${t.ts.split(' ')[1]}</td>
                        <td style="font-weight: 600;">${t.bot_id}</td>
                        <td>${t.side}</td>
                        <td>$${Math.round(t.price)}</td>
                        <td class="${pnlClass}">${t.pnl ? '$' + t.pnl.toFixed(2) : '-'}</td>
                    `;
                    history.appendChild(row);
                });

            } catch (e) {
                console.error("Dashboard update failed", e);
            }
        }

        setInterval(updateDashboard, 2000);
        updateDashboard();
    </script>
</body>
</html>
"""

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.get_data()).encode())
        else:
            super().do_GET()

    def get_data(self):
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            elif isinstance(obj, float):
                if obj != obj: return None
                return obj
            return obj

        # 1. Regime State
        regime = {}
        if os.path.exists(REGIME_FILE):
            try:
                with open(REGIME_FILE, "r") as f:
                    regime = json.load(f)
            except: pass
        
        # 2. Bot States
        bots = []
        for fpath in glob.glob(os.path.join(STATE_DIR, "*.json")):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    # Inject bot_id from filename if missing
                    if "bot_id" not in data:
                        data["bot_id"] = os.path.basename(fpath).replace(".json", "")
                    bots.append(data)
            except: pass
        
        # 3. Recent Trades (from DB)
        history = []
        try:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT ts, bot_id, side, price, realized_pnl FROM bot_status ORDER BY ts DESC LIMIT 10")
            for row in cur.fetchall():
                history.append({"ts": row[0], "bot_id": row[1], "side": row[2], "price": row[3], "pnl": row[4]})
            conn.close()
        except:
            pass

        return sanitize({
            "regime": regime,
            "bots": bots,
            "history": history
        })

with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
    print(f"Orchestrator Dashboard LIVE at http://localhost:{PORT}")
    httpd.serve_forever()
