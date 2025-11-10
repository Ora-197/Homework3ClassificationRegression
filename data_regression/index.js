
async function runExample1() {
    
  const x = new Float32Array(7);
  x[0] = parseFloat(document.getElementById('lr_box0').value) || 0;
  x[1] = parseFloat(document.getElementById('lr_box1').value) || 0;
  x[2] = parseFloat(document.getElementById('lr_box2').value) || 0;
  x[3] = parseFloat(document.getElementById('lr_box3').value) || 0;
  x[4] = parseFloat(document.getElementById('lr_box4').value) || 0;
  x[5] = parseFloat(document.getElementById('lr_box5').value) || 0;
  x[6] = parseFloat(document.getElementById('lr_box6').value) || 0;
 
  

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  try {
    const session = await ort.InferenceSession.create("./model_regression_LinearReg_movies.onnx?v=" + Date.now());
    const results = await session.run({ input: tensorX });
    const output = results.output.data; // Float32Array length 7

    
    // render here (output is in scope)
    const predictions = document.getElementById('predictions1');
    predictions.innerHTML = `<hr>Vote_average: <b>${output[0]}</b>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }


}






async function runExample2() {
    
  const x = new Float32Array(7);
  x[0] = parseFloat(document.getElementById('dl_box0').value) || 0;
  x[1] = parseFloat(document.getElementById('dl_box1').value) || 0;
  x[2] = parseFloat(document.getElementById('dl_box2').value) || 0;
  x[3] = parseFloat(document.getElementById('dl_box3').value) || 0;
  x[4] = parseFloat(document.getElementById('dl_box4').value) || 0;
  x[5] = parseFloat(document.getElementById('dl_box5').value) || 0;
  x[6] = parseFloat(document.getElementById('dl_box6').value) || 0;
 
  

  const tensorX = new ort.Tensor('float32', x, [1, 7]);

  try {
    const session = await ort.InferenceSession.create("./model_regression_DL_movies.onnx?v=" + Date.now());
    const results = await session.run({ input: tensorX });
    const output = results.output.data; // Float32Array length 7

    
    // render here (output is in scope)
    const predictions = document.getElementById('predictions2');
    predictions.innerHTML = `<hr>Vote_average: <b>${output[0]}</b>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  
  }
}

async function runExample3() {
    
    const x = new Float32Array(7);
    x[0] = parseFloat(document.getElementById('xgb_box0').value) || 0;
    x[1] = parseFloat(document.getElementById('xgb_box1').value) || 0;
    x[2] = parseFloat(document.getElementById('xgb_box2').value) || 0;
    x[3] = parseFloat(document.getElementById('xgb_box3').value) || 0;
    x[4] = parseFloat(document.getElementById('xgb_box4').value) || 0;
    x[5] = parseFloat(document.getElementById('xgb_box5').value) || 0;
    x[6] = parseFloat(document.getElementById('xgb_box6').value) || 0;
 
  

    const tensorX = new ort.Tensor('float32', x, [1, 7]);

    try {
        const session = await ort.InferenceSession.create("./xgboost_movies_vote_average_ort.onnx?v=" + Date.now());
        console.log("Outputs:", session.outputNames);
        const results = await session.run({ float_input: tensorX });
        const outputName = session.outputNames[0];  // le premier output
        const output = results[outputName].data;

        
        // render here (output is in scope)
        const predictions = document.getElementById('predictions3');
        predictions.innerHTML = `<hr>Vote_average: <b>${output[0]}</b>`;
    } catch (e) {
        console.error("ONNX runtime error:", e);
        alert("Error: " + e.message);
    }

    
}










