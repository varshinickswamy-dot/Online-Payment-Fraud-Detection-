function renderCharts(prob){

  const ctx = document.getElementById("barChart").getContext("2d");

  new Chart(ctx,{
    type:"bar",
    data:{
      labels:["Legit","Fraud"],
      datasets:[{
        label:"Probability %",
        data:[100-prob, prob],
        backgroundColor:["#22c55e","#ef4444"]
      }]
    },
    options:{
      responsive:false,
      scales:{
        y:{
          beginAtZero:true,
          max:100
        }
      }
    }
  });

}