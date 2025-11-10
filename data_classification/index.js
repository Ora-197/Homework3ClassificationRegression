

const index_to_genre = {
    0: 'Folk',
    1: 'Indie',
    2: 'Electronic',
    3: 'Comedy',
    4: 'Children’s Music',
    5: 'Hip-Hop',
    6: 'Jazz',
    7: 'Pop',
    8: 'Soundtrack',
    9: 'Rock'
};
async function runExample1() {
    
  const x = new Float32Array(10);
  x[0] = parseFloat(document.getElementById('mlp_box0').value) || 0;
  x[1] = parseFloat(document.getElementById('mlp_box1').value) || 0;
  x[2] = parseFloat(document.getElementById('mlp_box2').value) || 0;
  x[3] = parseFloat(document.getElementById('mlp_box3').value) || 0;
  x[4] = parseFloat(document.getElementById('mlp_box4').value) || 0;
  x[5] = parseFloat(document.getElementById('mlp_box5').value) || 0;
  x[6] = parseFloat(document.getElementById('mlp_box6').value) || 0;
  x[7] = parseFloat(document.getElementById('mlp_box7').value) || 0;
  x[8] = parseFloat(document.getElementById('mlp_box8').value) || 0;
  x[9] = parseFloat(document.getElementById('mlp_box9').value) || 0;
  

  const tensorX = new ort.Tensor('float32', x, [1, 10]);

  try {
    const session = await ort.InferenceSession.create("./model_classification_DL_musics_spotify.onnx?v=" + Date.now());
    const results = await session.run({ input: tensorX });
    const output = results.output.data; // Float32Array length 7

    const maxIndex = output.indexOf(Math.max(...output));
    const predicted_genre = index_to_genre[maxIndex];
    // render here (output is in scope)
    const predictions = document.getElementById('predictions1');
    predictions.innerHTML = `<hr>Predicted genre: <b>${predicted_genre}</b>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }


}






async function runExample2() {
    
  const x = new Float32Array(10);
  x[0] = parseFloat(document.getElementById('dl_box0').value) || 0;
  x[1] = parseFloat(document.getElementById('dl_box1').value) || 0;
  x[2] = parseFloat(document.getElementById('dl_box2').value) || 0;
  x[3] = parseFloat(document.getElementById('dl_box3').value) || 0;
  x[4] = parseFloat(document.getElementById('dl_box4').value) || 0;
  x[5] = parseFloat(document.getElementById('dl_box5').value) || 0;
  x[6] = parseFloat(document.getElementById('dl_box6').value) || 0;
  x[7] = parseFloat(document.getElementById('dl_box7').value) || 0;
  x[8] = parseFloat(document.getElementById('dl_box8').value) || 0;
  x[9] = parseFloat(document.getElementById('dl_box9').value) || 0;
  

  const tensorX = new ort.Tensor('float32', x, [1, 10]);

  try {
    const session = await ort.InferenceSession.create("./model_classification_DL_musics_spotify.onnx?v=" + Date.now());
    const results = await session.run({ input: tensorX });
    const output = results.output.data; // Float32Array length 7

    const maxIndex = output.indexOf(Math.max(...output));
    const predicted_genre = index_to_genre[maxIndex];
    // render here (output is in scope)
    const predictions = document.getElementById('predictions2');
    predictions.innerHTML = `<hr>Predicted genre: <b>${predicted_genre}</b>`;
  } catch (e) {
    console.error("ONNX runtime error:", e);
    alert("Error: " + e.message);
  }


    
}










