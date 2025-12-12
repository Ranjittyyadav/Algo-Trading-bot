// Trading Bot Dashboard - Enhanced JavaScript with Real-time Signals

// ✅ WEBSOCKET CONNECTION
const socket = io('http://0.0.0.0:5001');

// 🎨 Add mouse tracking for card glow effects
document.addEventListener('mousemove', (e) => {
    const cards = document.querySelectorAll('.status-card');
    cards.forEach(card => {
        const rect = card.getBoundingClientRect();
        const x = ((e.clientX - rect.left) / rect.width) * 100;
        const y = ((e.clientY - rect.top) / rect.height) * 100;
        card.style.setProperty('--mouse-x', `${x}%`);
        card.style.setProperty('--mouse-y', `${y}%`);
    });
});

let botData = {
    status: "DISCONNECTED",
    account_balance: 0,  // ✅ Start with 0, backend से आएगा
    floating_pnl: 0,
    open_positions: 0,
    total_trades: 0,
    winning_trades: 0,
    losing_trades: 0,
    win_rate: 0,
    active_positions: [],
    recent_trades: [],
    trading_signals: [],
    config: {}
};

// ✅ Listen for real balance data
socket.on('initial_data', (data) => {
    console.log('📊 Initial data received:', data);
    botData = {
        ...botData,
        ...data  // ✅ Merge incoming data (includes account_balance)
    };
    console.log(botData);
    renderDashboard();
    updateSystemStatus('✅ All systems connected', 'success');
});

socket.on('bot_update', (data) => {
    console.log('🔄 Update received:', data);
    
    // ✅ FIX: Don't overwrite trading_signals - preserve them
    const existingSignals = botData.trading_signals;
    const existingPositions = botData.active_positions;
    
    // Merge incoming data
    botData = {
        ...botData,      // Keep existing
        ...data,         // Apply new updates
        trading_signals: data.trading_signals || existingSignals,  // ✅ Preserve signals from backend
        active_positions: data.active_positions || existingPositions
    };
    
    renderDashboard();
    updateLastUpdate();
});


// ✅ NEW: Handle real-time trading signals
socket.on('trading_signal', (signal) => {
    console.log('📊 Trading signal:', signal);
    displaySignal(signal);
    playSignalSound();
});

socket.on('error', (error) => {
    console.error('❌ WebSocket error:', error);
    updateSystemStatus('❌ Connection error: ' + error, 'error');
});

// ✅ RENDER FUNCTIONS

function renderDashboard() {
    updateStatusBar();
    updatePerformanceMetrics();
    renderActivePositions();
    renderRecentTrades();
    renderSignals();
}

function updateStatusBar() {
    const statusEl = document.getElementById('status-text');
    const balanceEl = document.getElementById('account-balance');
    const pnlEl = document.getElementById('floating-pnl');
    const positionsEl = document.getElementById('open-positions');
    const winRateEl = document.getElementById('win-rate');
    const tradesEl = document.getElementById('total-trades');
    
    // ✅ FIX: Only update if elements exist
    if (statusEl) {
        statusEl.textContent = botData.status || 'OFFLINE';
    }
    if (balanceEl) {
        const balance = Number(botData.account_balance ?? 0);
        balanceEl.textContent = '$' + formatNumber(balance);
    }
    if (pnlEl) {
        const pnl = Number(botData.floating_pnl ?? 0);
        pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + formatNumber(pnl);
        pnlEl.style.color = pnl >= 0 ? '#10b981' : '#ef4444';
    }
    if (positionsEl) {
        positionsEl.textContent = botData.open_positions ?? 0;
    }
    if (winRateEl) {
        const wr = Number(botData.win_rate ?? 0);
        winRateEl.textContent = wr + '%';
        winRateEl.style.color = wr >= 30 ? '#10b981' : '#ef4444';
    }
    if (tradesEl) {
        tradesEl.textContent = botData.total_trades ?? 0;
    }
}


function updatePerformanceMetrics() {
        const totalTradesEl = document.getElementById('stat-total-trades');
    const winRateEl = document.getElementById('stat-win-rate');
    const winningEl = document.getElementById('stat-winning');
    const losingEl = document.getElementById('stat-losing');
    const floatingEl = document.getElementById('stat-floating-pnl');
    const balanceEl = document.getElementById('stat-balance');
    
    const floatingPnl = botData.floating_pnl || 0;
    // const floatingEl = document.getElementById('stat-floating-pnl');
    floatingEl.textContent = (floatingPnl >= 0 ? '+' : '') + '$' + formatNumber(floatingPnl);
    floatingEl.style.color = floatingPnl >= 0 ? '#10b981' : '#ef4444';
    
    document.getElementById('stat-balance').textContent = 
        '$' + formatNumber(botData.account_balance || 0);

        if (totalTradesEl) {
        totalTradesEl.textContent = botData.total_trades ?? 0;
    }
    if (winRateEl) {
        const wr = Number(botData.win_rate ?? 0);
        winRateEl.textContent = wr + '%';
    }
    if (winningEl) {
        winningEl.textContent = botData.winning_trades ?? 0;
    }
    if (losingEl) {
        losingEl.textContent = botData.losing_trades ?? 0;
    }
    if (floatingEl) {
        const floatingPnl = Number(botData.floating_pnl ?? 0);
        floatingEl.textContent = (floatingPnl >= 0 ? '+' : '') + '$' + formatNumber(floatingPnl);
        floatingEl.style.color = floatingPnl >= 0 ? '#10b981' : '#ef4444';
    }
    if (balanceEl) {
        const balance = Number(botData.account_balance ?? 0);
        balanceEl.textContent = '$' + formatNumber(balance);
    }
}

function renderActivePositions() {
    const container = document.getElementById('active-positions-container');
    
    // ✅ FIX: Add null check
    if (!container) {
        console.warn('⚠️ active-positions-container not found in HTML');
        return;
    }
    
    const positions = botData.active_positions || [];
    
    if (positions.length === 0) {
        container.innerHTML = '<div class="no-data">No active positions</div>';
        return;
    }
    
    container.innerHTML = positions.map(pos => `
        <div class="position-card">
            <div class="position-type ${pos.type.toLowerCase()}">${pos.type}</div>
            <div class="position-details">
                <strong>${pos.instrument}</strong> | ${pos.timeframe}<br>
                Entry: $${pos.entry.toFixed(2)}<br>
                SL: $${pos.sl.toFixed(2)} | TP: $${pos.tp.toFixed(2)}<br>
                P&L: <span style="color: ${pos.pnl >= 0 ? '#10b981' : '#ef4444'}">$${pos.pnl.toFixed(2)}</span>
            </div>
        </div>
    `).join('');
}


function renderRecentTrades() {
    const container = document.getElementById('recent-trades-container');
    
    // ✅ FIX: Add null check
    if (!container) {
        console.warn('⚠️ recent-trades-container not found in HTML');
        return;
    }
    
    const trades = botData.recent_trades || [];
    
    if (trades.length === 0) {
        container.innerHTML = '<div class="no-data">No recent trades</div>';
        return;
    }
    
    container.innerHTML = trades.slice(-10).reverse().map(trade => `
        <div class="trade-card">
            <div class="trade-type ${trade.type.toLowerCase()}">${trade.type}</div>
            <div class="trade-details">
                <strong>${trade.instrument}</strong> | ${trade.timeframe}<br>
                Entry: $${trade.entry.toFixed(2)}<br>
                Exit: $${(trade.exit_price || trade.tp).toFixed(2)}<br>
                P&L: <span style="color: ${trade.pnl >= 0 ? '#10b981' : '#ef4444'}">$${trade.pnl.toFixed(2)}</span>
            </div>
        </div>
    `).join('');
}


// ✅ NEW: Render Real-time Trading Signals

function renderSignals() {
    const container = document.getElementById('signals-container');
    
    // ✅ FIX: Add null check - if element doesn't exist, skip
    if (!container) {
        console.warn('⚠️ signals-container element not found in HTML');
        return;
    }
    
    const signals = botData.trading_signals || [];
    
    if (signals.length === 0) {
        container.innerHTML = `
            <div class="signal-item">
                <span class="signal-type candle">⏳ WAITING</span>
                <div class="signal-message">Waiting for trading signals...</div>
            </div>
        `;
        return;
    }
    
    // Show last 20 signals
    const visibleSignals = signals.slice(-20);
    
    container.innerHTML = visibleSignals.map((signal) => {
        const signalClass = signal.type === 'bullish' ? 'bullish' : 
                           signal.type === 'bearish' ? 'bearish' : 
                           'candle';
        
        let icon = '📊';
        if (signal.type === 'bullish') icon = '🟢';
        else if (signal.type === 'bearish') icon = '🔴';
        else if (signal.type === 'candle_complete') icon = '⏳';
        
        return `
            <div class="signal-item signal-${signalClass}">
                <span class="signal-type ${signalClass}">${icon} ${signal.type.toUpperCase()}</span>
                <div class="signal-message">${signal.message}</div>
                <div class="signal-time">${signal.timestamp}</div>
            </div>
        `;
    }).reverse().join('');
}



// ✅ NEW: Display signal in real-time

function displaySignal(signal) {
    // ✅ FIX: Add to botData.trading_signals instead of rendering directly
    if (!botData.trading_signals) {
        botData.trading_signals = [];
    }
    
    // Avoid duplicates - check if signal already exists
    const isDuplicate = botData.trading_signals.some(s => 
        s.timestamp_ms === signal.timestamp_ms && 
        s.type === signal.type &&
        s.instrument === signal.instrument
    );
    
    if (!isDuplicate) {
        botData.trading_signals.push(signal);
        // Keep only last 50 signals
        if (botData.trading_signals.length > 50) {
            botData.trading_signals.shift();
        }
    }
    
    // Re-render signals section
    renderSignals();
}


function updateSystemStatus(message, type = 'info') {
    const statusEl = document.getElementById('system-status');
    const alertClass = type === 'success' ? 'alert-success' : 
                       type === 'error' ? 'alert-error' : 'alert-warning';
    
    statusEl.innerHTML = `
        <div class="alert ${alertClass}">
            ${message}
        </div>
    `;
}

function updateLastUpdate() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    document.getElementById('last-update').textContent = 
        `Last updated: ${timeString}`;
}

// ✅ UTILITY FUNCTIONS

function formatNumber(num) {
    return Math.abs(num).toFixed(2).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

function refreshPositions() {
    console.log('🔄 Refreshing positions...');
    socket.emit('bot_update');
}

function playSignalSound() {
    // 🎵 Enhanced notification sound with better tones
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // 🎼 Play a pleasant chord progression
        const notes = [523.25, 659.25, 783.99]; // C5, E5, G5
        let time = audioContext.currentTime;
        
        notes.forEach((freq, i) => {
            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();
            osc.connect(gain);
            gain.connect(audioContext.destination);
            
            osc.frequency.value = freq;
            osc.type = 'sine';
            
            gain.gain.setValueAtTime(0.15, time);
            gain.gain.exponentialRampToValueAtTime(0.01, time + 0.3);
            
            osc.start(time + (i * 0.1));
            osc.stop(time + (i * 0.1) + 0.3);
        });
        
        // ✨ Add visual notification
        showNotificationFlash();
    } catch (e) {
        console.log('Audio notification skipped');
    }
}

// ✨ Visual notification flash
function showNotificationFlash() {
    const flash = document.createElement('div');
    flash.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 16px 24px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.9), rgba(139, 92, 246, 0.9));
        color: white;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.4);
        z-index: 9999;
        animation: slideInRight 0.3s ease-out, fadeOut 0.3s ease-in 2.7s;
        font-weight: 600;
        backdrop-filter: blur(10px);
    `;
    flash.textContent = '📊 New Signal Received!';
    document.body.appendChild(flash);
    
    setTimeout(() => flash.remove(), 3000);
}

// Add this to see what data you're receiving
socket.on('initial_data', (data) => {
    console.log("Initial data received:", data);
    console.log("Account balance:", data.account_balance);
    console.log("Floating P&L:", data.floating_pnl);
});


// ✅ INITIAL LOAD
console.log('🚀 Enhanced Dashboard initialized');
console.log('📡 Connecting to WebSocket: http://localhost:5001');

// Request update on page load
setTimeout(() => {
    socket.emit('bot_update');
}, 1000);

const originalOn = socket.on.bind(socket);
socket.on = function(eventName, callback) {
    console.log(`✅ Listening for event: ${eventName}`);
    return originalOn(eventName, callback);
};

// Periodic update request (every 5 seconds)
setInterval(() => {
    if (socket.connected) {
        socket.emit('bot_update');
    }
}, 5001);
