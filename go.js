const canvas = document.getElementById("goBoard");
const ctx = canvas.getContext("2d");

const N = 9;            // board size
const size = canvas.width;
const cell = size / (N + 1);

let board = Array.from({ length: N }, () => Array(N).fill("O"));
let turn = "B";
let isProcessing = false;

/* ------------------------- Drawing Functions ------------------------- */

// Draw board grid + stones
function drawBoard() {
  ctx.clearRect(0, 0, size, size);

  // grid
  ctx.strokeStyle = "#000";
  for (let i = 1; i <= N; i++) {
    ctx.beginPath();
    ctx.moveTo(cell, i * cell);
    ctx.lineTo(N * cell, i * cell);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(i * cell, cell);
    ctx.lineTo(i * cell, N * cell);
    ctx.stroke();
  }

  // draw star points (hoshi)
  drawStarPoints();

  // stones
  for (let row = 0; row < N; row++) {
    for (let col = 0; col < N; col++) {
      if (board[row][col] == "B" || board[row][col] == "W") drawStone(col, row, board[row][col]);
    }
  }
}

function drawStarPoints() {
  const points = (N === 19) ? [4, 10, 16] : (N === 13) ? [4, 7, 10] : [3, 7];
  ctx.fillStyle = "#000";

  for (let py of points) {
    for (let px of points) {
      ctx.beginPath();
      ctx.arc(px * cell, py * cell, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

function drawStone(col, row, color) {
  ctx.beginPath();
  ctx.arc((col + 1) * cell, (row + 1) * cell, cell * 0.45, 0, Math.PI * 2);
  if(color == "B") {
    ctx.fillStyle = "black";
  } else {
    ctx.fillStyle = "white";
  }
  ctx.fill();
}

// =========================
// Backend Communication
// =========================
async function sendMoveToServer(row, col) {
    if(turn === "B" && isProcessing) return;
    console.log("Changed to true", turn);
    isProcessing = true;
    try {
        console.log(board[row][col], turn, row, col);
        console.log("76", turn);
        const response = await fetch("http://localhost:8000/move", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ turn: turn, col: col, row: row})
        });
        console.log("82", turn);
        if (!response.ok) {
            console.error("Server error:", await response.text());
            return;
        }
        console.log("87", turn);
        const result = await response.json();

        board = result.board;       // backend authoritative
        console.log("91", turn);
        drawBoard();
    }
    catch (e) {
        console.error("Connection error:", e);
    }
    finally {
      console.log("Finally", turn);   
      if(turn === "W") {
        console.log("Changed to false");
        isProcessing = false;
      }
      
    }
}

// =========================
// Mouse Input
// =========================
canvas.addEventListener("click", async (e) => {
    console.log(isProcessing);
    if (isProcessing) return;
    const rect = canvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Find nearest board point
    const col= Math.round(clickX / cell - 1);
    const row = Math.round(clickY / cell - 1);
    if (col< 0 || col>= N || row < 0 || row >= N) return;
    console.log(board[row][col]);
    if(board[row][col] != "O") return;
    await sendMoveToServer(row, col);
    turn = turn === "B" ? "W" : "B";
    drawBoard();
    console.log("turn", turn);
    await sendMoveToServer(row, col);
    turn = turn === "B" ? "W" : "B";
});

/* ------------------------- Run the UI ------------------------- */
drawBoard();