	function showCloudView(data){
  // create a tag (word) cloud chart
  $('#container').remove(); // this is my <canvas> element
  $('#parent-container').append('<div id="container"></div>');

  var chart = anychart.tagCloud(data);
  // set an array of angles at which the words will be laid out
  chart.angles([0])
  
  chart.container("container");
  chart.draw();
  			setTimeout(function(){ 

  $('.anychart-credits-text').remove();

  $('.anychart-credits-logo').remove();
  			
			}, 200);
	}