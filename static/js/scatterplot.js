"use strict";

(function (Utils) {
  const chartjsUrl = "https://cdn.jsdelivr.net/npm/chart.js/dist/chart.js";
  const dateFnsUrl =
    "https://cdnjs.cloudflare.com/ajax/libs/chartjs-adapter-moment/1.0.0/chartjs-adapter-moment.min.js";
  const localUrl =
    "https://cdn.jsdelivr.net/npm/chartjs-chart-matrix/dist/chartjs-chart-matrix.js";
  const remoteUrl =
    "https://cdn.jsdelivr.net/npm/chartjs-chart-matrix/dist/chartjs-chart-matrix.js";

  function addScript(url, done, error) {
    const head = document.getElementsByTagName("head")[0];
    const script = document.createElement("script");
    script.type = "text/javascript";
    script.onreadystatechange = function () {
      if (this.readyState === "complete") {
        done();
      }
    };
    script.onload = done;
    script.onerror = error;
    script.src = url;
    head.appendChild(script);
    return true;
  }

  function loadError() {
    const msg = document.createTextNode("Error loading chartjs-chart-matrix");
    document.body.appendChild(msg);
    return true;
  }

  Utils.load = function (done) {
    addScript(
      chartjsUrl,
      () => {
        addScript(
          dateFnsUrl,
          () => {
            addScript(localUrl, done, (event) => {
              event.preventDefault();
              event.stopPropagation();
              addScript(remoteUrl, done, loadError);
            });
          },
          loadError
        );
      },
      loadError
    );
  };
})((window.Utils = window.Utils || {}));

function isoDayOfWeek(dt) {
  let wd = dt.getDay(); // 0..6, from sunday
  wd = ((wd + 6) % 7) + 1; // 1..7 from monday
  return ""+wd; 
}
function generateData2() {
  const data = [];
  const end = new Date(2021, 3, 15);
  let dt = new Date(2021, 1, 1);
  while (dt <= end) {
    const iso = dt.toISOString().substr(0, 10);
    data.push({
      x: iso,
      y: isoDayOfWeek(dt),
      v: Math.random() * 50
    });
    dt = new Date(dt.setDate(dt.getDate() + 1));
  }
  data.splice(5,1);
  data.splice(12,1);
  data.splice(10,1);
  return data;
}

Utils.load(() => {
  Chart.defaults.fontSize = 9;
  const ctx2 = document.getElementById("chart-area2").getContext("2d");
  const data = {
  datasets: [{
    label: 'Similarity Plot View',//scoreData,
    data: scoreData,/*[
    {x: 'A', y: 'A', v: 1},
    {x: 'A', y: 'B', v: 0.2},
    {x: 'A', y: 'C', v: 0.5},
    {x: 'B', y: 'A', v: 0.2},
    {x: 'B', y: 'B', v: 1},
    {x: 'B', y: 'C', v: 0.7},
    {x: 'C', y: 'A', v: 0.5},
    {x: 'C', y: 'B', v: 0.7},
    {x: 'C', y: 'C', v: 1}
    ],*/
    backgroundColor(context) {
      const value = context.dataset.data[context.dataIndex].v;
      const alpha = value;//((value*100) - 5) / 100;
      return Chart.helpers.color('yellow').alpha(alpha).rgbString();
    },
    borderColor(context) {
      const value = context.dataset.data[context.dataIndex].v;
      const alpha = value;//((value*100) - 5) / 100;
      return Chart.helpers.color('yellow').alpha(alpha).rgbString();
    },
    borderWidth: 1,
    width: ({chart}) => (chart.chartArea || {}).width / recommendationsList.length - 1,
    height: ({chart}) =>(chart.chartArea || {}).height / recommendationsList.length - 1
  }]
};
  window.myMatrix2 =  new Chart(ctx2, 
	{
	  type: 'matrix',
	  data: data,
		  options: {
			plugins: {
			  legend: false,
			  tooltip: {
				callbacks: {
				  title() {
					return '';
				  },
				  label(context) {
					const v = context.dataset.data[context.dataIndex];
					return ['x: ' + v.x, 'y: ' + v.y, 'TF-IDF score: ' + v.v];
				  }
				}
			  }
			},
			scales: {
			  x: {
				type: 'category',
				labels: scatterlabels,//['A', 'B', 'C'],
				ticks: {
				  display: true
				},
				grid: {
				  display: false
				}
			  },
			  y: {
				type: 'category',
				labels: scatterlabels,//['A', 'B', 'C'],
				offset: true,
				ticks: {
				  display: true
				},
				grid: {
				  display: false
				}
			  }
			}
		}
	}
);
  
  
});
