const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let drawing = false;
canvas.addEventListener("mousedown", e => {
  drawing = true;
  ctx.beginPath();
  ctx.moveTo(e.offsetX, e.offsetY);
});
canvas.addEventListener("mousemove", e => {
  if (!drawing) return;
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
});
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseleave", () => drawing = false);

document.getElementById("clear").addEventListener("click", () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("result").textContent = "";
});

let session;
async function loadModel() {
  session = await ort.InferenceSession.create("model.onnx", {
    executionProviders: ["wasm"]
  });
  console.log("Modèle chargé");
}
loadModel();

function preprocessCanvas(canvas) {
  const tmp = document.createElement("canvas");
  tmp.width = 28;
  tmp.height = 28;
  const tmpCtx = tmp.getContext("2d");
  tmpCtx.drawImage(canvas, 0, 0, 28, 28);
  const data = tmpCtx.getImageData(0,0,28,28).data;

  const input = new Float32Array(28*28);
  for(let i=0;i<28*28;i++){
    const r = data[i*4];
    input[i] = (255 - r)/255; 
  }
  return new ort.Tensor("float32", input, [1,1,28,28]);
}


async function predict() {
    if(!session) return alert("Modèle pas encore chargé !");
    const inputTensor = preprocessCanvas(canvas);
  

    const feeds = { x: inputTensor };  
  
    const results = await session.run(feeds);
  

    const outputKey = Object.keys(results)[0];
    const output = results[outputKey].data;
  
   
    let maxIdx = 0;
    for(let i=1;i<output.length;i++){
      if(output[i] > output[maxIdx]) maxIdx = i;
    }
  
    document.getElementById("result").textContent = "Résultat : " + maxIdx;
  }
  

document.getElementById("predict").addEventListener("click", predict);
