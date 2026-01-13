// Game state
let sessionId = null;
let playerColor = 'black';
let useRenju = true;
let useAI = true;
let gameActive = false;
let boardState = null;
let moveHistory = [];  // Track move order: [{x, y, color, moveNumber}, ...]
let forbiddenMoves = [];  // Track forbidden moves for current turn
let currentTurn = 'black';  // Track whose turn it is
let showMoveNumbers = false;
let winrateHistory = []; // [{black: 0..1, white: 0..1}, ...]
const BOARD_SIZE = 15;
const CELL_SIZE = 40;

// Canvas elements
const canvas = document.getElementById('gameBoard');
const ctx = canvas.getContext('2d');

// UI elements
const gameInfo = document.getElementById('gameInfo');
const message = document.getElementById('message');
const newGameBtn = document.getElementById('newGameBtn');
const blackBtn = document.getElementById('blackBtn');
const whiteBtn = document.getElementById('whiteBtn');
const useRenjuCheckbox = document.getElementById('useRenjuCheckbox');
const useAICheckbox = document.getElementById('useAICheckbox');
const showNumbersCheckbox = document.getElementById('showNumbersCheckbox');
const difficultySlider = document.getElementById('difficultySlider');
const difficultyValue = document.getElementById('difficultyValue');

const winrateCanvas = document.getElementById('winrateChart');
const winrateCtx = winrateCanvas.getContext('2d');

const THINKING_TEXT = 'AI가 생각 중입니다...';
let isThinking = false;

// Initialize
drawEmptyBoard();
syncDifficultyUI();
syncWinrateVisibility();
clearWinrateChart();

showNumbersCheckbox.checked = false;
showNumbersCheckbox.addEventListener('change', () => {
    showMoveNumbers = showNumbersCheckbox.checked;
    drawBoard(boardState, moveHistory, forbiddenMoves);
});

difficultySlider.addEventListener('input', () => {
    syncDifficultyUI();
});

useAICheckbox.addEventListener('change', () => {
    syncWinrateVisibility();
});

// Color selection
function selectColor(color) {
    playerColor = color;
    blackBtn.classList.toggle('selected', color === 'black');
    whiteBtn.classList.toggle('selected', color === 'white');
}

// Draw empty board
function drawEmptyBoard() {
    ctx.fillStyle = '#daa520';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    
    for (let i = 0; i < BOARD_SIZE; i++) {
        // Vertical lines
        ctx.beginPath();
        ctx.moveTo(CELL_SIZE * (i + 0.5), CELL_SIZE * 0.5);
        ctx.lineTo(CELL_SIZE * (i + 0.5), CELL_SIZE * (BOARD_SIZE - 0.5));
        ctx.stroke();
        
        // Horizontal lines
        ctx.beginPath();
        ctx.moveTo(CELL_SIZE * 0.5, CELL_SIZE * (i + 0.5));
        ctx.lineTo(CELL_SIZE * (BOARD_SIZE - 0.5), CELL_SIZE * (i + 0.5));
        ctx.stroke();
    }
    
    // Draw star points
    const starPoints = [3, 11];
    ctx.fillStyle = '#000';
    for (let x of starPoints) {
        for (let y of starPoints) {
            ctx.beginPath();
            ctx.arc(CELL_SIZE * (x + 0.5), CELL_SIZE * (y + 0.5), 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }
    ctx.beginPath();
    ctx.arc(CELL_SIZE * (7 + 0.5), CELL_SIZE * (7 + 0.5), 3, 0, 2 * Math.PI);
    ctx.fill();
}

// Draw board with pieces
function drawBoard(board, history = moveHistory, forbidden = forbiddenMoves) {
    drawEmptyBoard();
    
    // Draw forbidden moves (red X marks)
    if (useRenju && forbidden) {
        for (let move of forbidden) {
            drawForbiddenMark(move.x, move.y, move.type || '');
        }
    }
    
    // Draw pieces with move numbers
    for (let move of history) {
        drawPiece(move.x, move.y, move.color, showMoveNumbers ? move.moveNumber : null);
    }

    if (!showMoveNumbers && history.length > 0) {
        const last = history[history.length - 1];
        drawLastMoveMarker(last.x, last.y);
    }
}

// Draw a single piece
function drawPiece(x, y, color, moveNumber = null) {
    const centerX = CELL_SIZE * (x + 0.5);
    const centerY = CELL_SIZE * (y + 0.5);
    const radius = CELL_SIZE * 0.4;
    
    // Shadow
    ctx.beginPath();
    ctx.arc(centerX + 2, centerY + 2, radius, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.fill();
    
    // Main stone
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    
    if (color === 1) { // Black
        const gradient = ctx.createRadialGradient(centerX - 5, centerY - 5, 0, centerX, centerY, radius);
        gradient.addColorStop(0, '#666');
        gradient.addColorStop(1, '#000');
        ctx.fillStyle = gradient;
    } else { // White
        const gradient = ctx.createRadialGradient(centerX - 5, centerY - 5, 0, centerX, centerY, radius);
        gradient.addColorStop(0, '#fff');
        gradient.addColorStop(1, '#ddd');
        ctx.fillStyle = gradient;
    }
    ctx.fill();
    
    // Border
    ctx.strokeStyle = color === 1 ? '#000' : '#999';
    ctx.lineWidth = 1;
    ctx.stroke();
    
    // Draw move number
    if (moveNumber !== null) {
        ctx.font = 'bold 14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        // Check if this is the last move
        const isLastMove = moveNumber === moveHistory.length;
        ctx.fillStyle = isLastMove ? '#ff0000' : (color === 1 ? '#fff' : '#000');
        ctx.fillText(moveNumber.toString(), centerX, centerY);
    }
}

// Draw last move marker (red square dot)
function drawLastMoveMarker(x, y) {
    const centerX = CELL_SIZE * (x + 0.5);
    const centerY = CELL_SIZE * (y + 0.5);
    const size = 8;
    ctx.fillStyle = '#ff0000';
    ctx.fillRect(centerX - size / 2, centerY - size / 2, size, size);
}

// Draw forbidden mark (red X)
function drawForbiddenMark(x, y, type = '') {
    const centerX = CELL_SIZE * (x + 0.5);
    const centerY = CELL_SIZE * (y + 0.5);
    const size = CELL_SIZE * 0.3;
    
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    
    // Draw X
    ctx.beginPath();
    ctx.moveTo(centerX - size, centerY - size);
    ctx.lineTo(centerX + size, centerY + size);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(centerX + size, centerY - size);
    ctx.lineTo(centerX - size, centerY + size);
    ctx.stroke();
    
    // Draw type text inside X
    if (type) {
        ctx.font = 'bold 12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ff0000';
        // 2px stroke
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#ffffff';
        ctx.strokeText(type, centerX, centerY);
        ctx.fillText(type, centerX, centerY);
    }
}

// Start new game
async function startNewGame() {
    showThinkingMessage('게임을 시작하는 중...');
    setMessage('', 'info');
    moveHistory = [];  // Reset move history
    forbiddenMoves = [];  // Reset forbidden moves
    winrateHistory = [];
    clearWinrateChart();
    
    // Get checkbox states
    useRenju = useRenjuCheckbox.checked;
    useAI = useAICheckbox.checked;
    syncWinrateVisibility();

    const mctsCount = clampInt(parseInt(difficultySlider.value, 10), 400, 1200);
    
    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                player_color: playerColor,
                use_renju: useRenju,
                use_ai: useAI,
                mcts_count: mctsCount
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            sessionId = data.session_id;
            boardState = data.board;
            gameActive = true;
            forbiddenMoves = data.forbidden_moves || [];
            
            // Set initial turn
            currentTurn = 'black';
            
            // If AI made first move, add it to history
            if (data.ai_move) {
                const aiColorNum = playerColor === 'black' ? 2 : 1;
                moveHistory.push({
                    x: data.ai_move.x,
                    y: data.ai_move.y,
                    color: aiColorNum,
                    moveNumber: 1
                });
                currentTurn = playerColor;  // Now it's player's turn
                setMessage(`AI가 (${String.fromCharCode(65 + data.ai_move.x)}, ${15 - data.ai_move.y})에 두었습니다`, 'info');
            }

            if (useAI && data.winrate) {
                pushWinrate(data.winrate);
            }
            
            drawBoard(boardState, moveHistory, forbiddenMoves);
            canvas.classList.remove('disabled');
            newGameBtn.disabled = false;
            updateGameInfo();
            
            if (data.is_done) {
                handleGameEnd(data.winner);
            }
        } else {
            setMessage('게임 시작 실패: ' + data.error, 'error');
        }
    } catch (error) {
        setMessage('서버 연결 오류: ' + error.message, 'error');
    } finally {
        hideThinkingMessage();
    }
}

// Handle canvas click
canvas.addEventListener('click', async (event) => {
    if (!gameActive) return;
    
    const rect = canvas.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;
    
    const x = Math.round(clickX / CELL_SIZE - 0.5);
    const y = Math.round(clickY / CELL_SIZE - 0.5);
    
    if (x < 0 || x >= BOARD_SIZE || y < 0 || y >= BOARD_SIZE) return;
    
    // Check if position is empty
    const index = x + y * BOARD_SIZE;
    if (boardState[index] !== 0) {
        setMessage('이미 돌이 놓인 위치입니다', 'error');
        return;
    }
    
    await makeMove(x, y);
});

// Make a move
async function makeMove(x, y) {
    gameActive = false;
    canvas.classList.add('disabled');
    
    // Determine whose turn it is
    const currentPlayerColorNum = currentTurn === 'black' ? 1 : 2;
    
    // Add current player's move to history and draw immediately
    moveHistory.push({x, y, color: currentPlayerColorNum, moveNumber: moveHistory.length + 1});
    drawBoard(boardState, moveHistory, forbiddenMoves);
    
    // Show AI thinking message in message div
    if (useAI) showThinkingMessage(THINKING_TEXT);
    
    try {
        const response = await fetch('/api/move', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: sessionId,
                x: x,
                y: y
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            boardState = data.board;
            forbiddenMoves = data.forbidden_moves || [];

            if (data.ai_move) {
                // AI made a move
                const aiColorNum = currentTurn === 'black' ? 2 : 1;
                moveHistory.push({
                    x: data.ai_move.x,
                    y: data.ai_move.y,
                    color: aiColorNum,
                    moveNumber: moveHistory.length + 1
                });
                drawBoard(boardState, moveHistory, forbiddenMoves);
                setMessage(`AI가 (${String.fromCharCode(65 + data.ai_move.x)}, ${15 - data.ai_move.y})에 두었습니다`, 'info');

                if (useAI && data.winrate) {
                    pushWinrate(data.winrate);
                }

                if (data.is_done) {
                    // IMPORTANT: show win message only after AI stone is drawn
                    handleGameEnd(data.winner);
                    return;
                }

                gameActive = true;
                canvas.classList.remove('disabled');
                return;
            }

            if (useAI && data.winrate) {
                pushWinrate(data.winrate);
            }

            if (data.is_done) {
                // Player just won; stone already drawn locally.
                handleGameEnd(data.winner);
            } else {
                // No AI move (player vs player mode)
                // Switch turn
                currentTurn = currentTurn === 'black' ? 'white' : 'black';
                drawBoard(boardState, moveHistory, forbiddenMoves);
                updateGameInfo();
                gameActive = true;
                canvas.classList.remove('disabled');
            }
        } else {
            // If move was invalid, remove it from history
            moveHistory.pop();
            drawBoard(boardState, moveHistory, forbiddenMoves);
            setMessage('잘못된 수입니다: ' + data.error, 'error');
            gameActive = true;
            canvas.classList.remove('disabled');
        }
    } catch (error) {
        // If error occurred, remove the move from history
        moveHistory.pop();
        drawBoard(boardState, moveHistory, forbiddenMoves);
        setMessage('서버 연결 오류: ' + error.message, 'error');
        gameActive = true;
        canvas.classList.remove('disabled');
    } finally {
        hideThinkingMessage();
    }
}

// Handle game end
function handleGameEnd(winner) {
    gameActive = false;
    canvas.classList.add('disabled');
    newGameBtn.disabled = false;
    
    if (winner === 'draw') {
        setMessage('무승부입니다!', 'success');
        gameInfo.textContent = '게임 종료 - 무승부';
    } else {
        const winnerText = (winner === 'black') ? '⚫ 흑' : '⚪ 백';
        setMessage(`${winnerText}의 승리입니다!`, 'success');
        gameInfo.textContent = `게임 종료 - ${winnerText} 승리`;
    }
}

// Update game info
function updateGameInfo() {
    if (useAI) {
        const colorText = playerColor === 'black' ? '⚫ 흑' : '⚪ 백';
        gameInfo.textContent = `${colorText}의 차례입니다`;
    } else {
        const colorText = currentTurn === 'black' ? '⚫ 흑' : '⚪ 백';
        gameInfo.textContent = `${colorText}의 차례입니다`;
    }
}

function showThinkingMessage(text) {
    isThinking = true;
    setMessage(text, 'info');
}

function hideThinkingMessage() {
    if (isThinking && message.textContent === THINKING_TEXT) {
        setMessage('', 'info');
    }
    isThinking = false;
}

// Set message
function setMessage(text, type = 'info') {
    message.textContent = text;
    message.className = 'message ' + type;
}

function clampInt(value, min, max) {
    if (!Number.isFinite(value)) return min;
    return Math.max(min, Math.min(max, value));
}

function syncDifficultyUI() {
    const mctsCount = clampInt(parseInt(difficultySlider.value, 10), 400, 1200);
    difficultySlider.value = mctsCount;
    difficultyValue.textContent = String(mctsCount);
    difficultySlider.disabled = !useAICheckbox.checked;
}

function syncWinrateVisibility() {
    const visible = useAICheckbox.checked;
    winrateCanvas.style.display = visible ? 'block' : 'none';
}

function pushWinrate(winrate) {
    const black = typeof winrate.black === 'number' ? winrate.black : null;
    const white = typeof winrate.white === 'number' ? winrate.white : null;
    if (black === null || white === null) return;
    winrateHistory.push({
        black: Math.max(0, Math.min(1, black)),
        white: Math.max(0, Math.min(1, white))
    });
    drawWinrateChart();
}

function clearWinrateChart() {
    winrateCtx.clearRect(0, 0, winrateCanvas.width, winrateCanvas.height);
}

function drawWinrateChart() {
    const w = winrateCanvas.width;
    const h = winrateCanvas.height;
    winrateCtx.clearRect(0, 0, w, h);

    // Background
    winrateCtx.fillStyle = '#ffffff';
    winrateCtx.fillRect(0, 0, w, h);

    // Border
    winrateCtx.strokeStyle = '#333';
    winrateCtx.lineWidth = 1;
    winrateCtx.strokeRect(0.5, 0.5, w - 1, h - 1);

    // Title
    winrateCtx.fillStyle = '#555';
    winrateCtx.font = 'bold 12px Arial';
    winrateCtx.textAlign = 'left';
    winrateCtx.textBaseline = 'top';
    winrateCtx.fillText('예상 승률 (흑/백)', 8, 6);

    if (winrateHistory.length === 0) return;

    // Latest percentage text
    const last = winrateHistory[winrateHistory.length - 1];
    const blackPct = Math.round(last.black * 100);
    const whitePct = Math.round(last.white * 100);
    winrateCtx.font = 'bold 12px Arial';
    winrateCtx.textAlign = 'right';
    winrateCtx.textBaseline = 'top';
    winrateCtx.fillStyle = '#111';
    winrateCtx.fillText(`흑 ${blackPct}% : 백 ${whitePct}%`, w - 8, 6);

    const leftPad = 6;
    const rightPad = 6;
    const topPad = 22;
    const bottomPad = 10;
    const chartW = w - leftPad - rightPad;
    const chartH = h - topPad - bottomPad;

    const n = winrateHistory.length;
    const xAt = (i) => (n === 1 ? leftPad + chartW / 2 : leftPad + (chartW * i) / (n - 1));
    const yAtBlack = (black) => topPad + chartH * (1 - black);

    // Fill black area (bottom to line)
    winrateCtx.beginPath();
    winrateCtx.moveTo(xAt(0), topPad + chartH);
    for (let i = 0; i < n; i++) {
        winrateCtx.lineTo(xAt(i), yAtBlack(winrateHistory[i].black));
    }
    winrateCtx.lineTo(xAt(n - 1), topPad + chartH);
    winrateCtx.closePath();
    winrateCtx.fillStyle = 'rgba(0, 0, 0, 1.0)';
    winrateCtx.fill();

    // Fill white area (top to line)
    winrateCtx.beginPath();
    winrateCtx.moveTo(xAt(0), topPad);
    for (let i = 0; i < n; i++) {
        winrateCtx.lineTo(xAt(i), yAtBlack(winrateHistory[i].black));
    }
    winrateCtx.lineTo(xAt(n - 1), topPad);
    winrateCtx.closePath();
    winrateCtx.fillStyle = 'rgba(230, 230, 230, 1.0)';
    winrateCtx.fill();

    // Line
    winrateCtx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = xAt(i);
        const y = yAtBlack(winrateHistory[i].black);
        if (i === 0) winrateCtx.moveTo(x, y);
        else winrateCtx.lineTo(x, y);
    }
    winrateCtx.strokeStyle = '#111';
    winrateCtx.lineWidth = 2;
    winrateCtx.stroke();
}
