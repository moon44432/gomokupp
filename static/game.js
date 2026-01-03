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
const BOARD_SIZE = 15;
const CELL_SIZE = 40;

// Canvas elements
const canvas = document.getElementById('gameBoard');
const ctx = canvas.getContext('2d');

// UI elements
const gameInfo = document.getElementById('gameInfo');
const message = document.getElementById('message');
const loading = document.getElementById('loading');
const newGameBtn = document.getElementById('newGameBtn');
const blackBtn = document.getElementById('blackBtn');
const whiteBtn = document.getElementById('whiteBtn');
const useRenjuCheckbox = document.getElementById('useRenjuCheckbox');
const useAICheckbox = document.getElementById('useAICheckbox');

// Initialize
drawEmptyBoard();

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
        drawPiece(move.x, move.y, move.color, move.moveNumber);
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
    showLoading('게임을 시작하는 중...');
    setMessage('', 'info');
    moveHistory = [];  // Reset move history
    forbiddenMoves = [];  // Reset forbidden moves
    
    // Get checkbox states
    useRenju = useRenjuCheckbox.checked;
    useAI = useAICheckbox.checked;
    
    try {
        const response = await fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                player_color: playerColor,
                use_renju: useRenju,
                use_ai: useAI
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
        hideLoading();
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
    
    // Show AI thinking message if AI is next
    if (useAI) {
        showLoading('AI가 생각하는 중...');
    }
    
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
            
            if (data.is_done) {
                handleGameEnd(data.winner);
            } else if (data.ai_move) {
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
                gameActive = true;
                canvas.classList.remove('disabled');
                
                if (data.is_done) {
                    handleGameEnd(data.winner);
                }
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
        hideLoading();
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

// Show/hide loading
function showLoading(text) {
    loading.textContent = text;
}

function hideLoading() {
    loading.textContent = '';
}

// Set message
function setMessage(text, type = 'info') {
    message.textContent = text;
    message.className = 'message ' + type;
}
