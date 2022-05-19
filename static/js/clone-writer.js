//implement function to display content in code mirror windows
var mainEditor;
var recomendEditor1;
var recomendEditor2;
var recomendEditor3;
var recomendEditor4;
var recomendEditor5;
var recomendEditor6;
var recomendEditorList=[]
var identifierList;
var scoreList;
var recommendationsList;	
var checkBoxList;
var scoreData=[];
var scatterlabels=[];

function displayMainWindowContent() {
	mainEditor = CodeMirror.fromTextArea(document.getElementById("java-code"), {
        lineNumbers: true,
        matchBrackets: true,
        mode: "text/x-java",
		lineNumbers: true,
                    matchBrackets: true,
                    extraKeys: {
                        "Ctrl-Space": "autocomplete"
                    },
                    gutters: ["CodeMirror-lint-markers"],
                    lint: true,
      });
		      mainEditor.setSize(null, '100%');
}
function getCodeTillCursor(){
	var stringToTest=""
	for(var i=0;i<=mainEditor.doc.getCursor().line;i++){
		if(i==mainEditor.doc.getCursor().line){
			var ch = mainEditor.doc.getCursor().ch;       //Cursor character
			// Select all characters before cursor
			stringToTest = stringToTest+mainEditor.doc.getLine(i).substr(0, ch);
		}
		else{
			stringToTest = stringToTest+mainEditor.doc.getLine(i);			
		}
	}
	
	return stringToTest;
}

function autoFormat(editor) {
         		var totalLines = editor.lineCount();  
                 editor.autoFormatRange({line:0, ch:0}, {line:totalLines});
}

function searchCode(){
	getPredictedClones($('#searchtext').val(),1)
}

function getPredictedClones(inputString,isSearchScreen) {
	  var topp=$('#topId').val();
	var tokenSize=$('#tokensId').val()
var code=getCodeTillCursor();

				if(isSearchScreen==1){
					$('#searchView').modal('toggle');
				}
			 $('#appLoader').fadeIn();		
		 $.ajax({
			type : 'POST',
			url : '/getRecommendations',
			data : {
				inputString : code,
				topp:topp,
				tokenSize:tokenSize,
				kvalue:$('#k-value').val()
				},
			success: function (result) {
				console.log(result);
				if(result.recommendations){			
							identifierList=result.recommendedIdentifiers;
							scoreList=result.scores;			
							recommendationsList=result.recommendations;
							  $('#recommendWindows').children().remove();
							  
							  createRecommendWindows();
							  
						  for(var i=0;i<recomendEditorList.length;i++){
							recomendEditorList[i].setValue(result.recommendations[i]);  
							autoFormat(recomendEditorList[i]);
						  }
						toastr.success('Recommended clone methods have been successfully retrieved!');

				}
				else{
						toastr.warning('No recommended clone methods can be retrieved!');					
				}
						  $('#appLoader').fadeOut();
			},
			error: function (error) {
				console.log(error);
				$('#searchView').modal('toggle');
   			  $('#appLoader').fadeOut();
				toastr["error"]("Something is wrong!");
			}
		});
	  
}
//display textual search results
function getDictionaryArray(scores){
	tempList = [];

for (i = 0; i < recommendLabelsList.length; i++) {
  tempList[i] = recommendLabelsList[i];
}

tempList.push("R"+(recomendEditorList.length+1))	
var arr = tempList.map(function(item,i,arr) {
    var tmp = arr.map(function(_item) { return [item, _item];});//if( item != _item) 
    return tmp.splice(tmp.indexOf(undefined),1), tmp;
});
arr.splice(-1);
var matrix=[]
for (var i=0;i<arr.length;i++)
{
	for(var j=0;j<arr[i].length;j++){
		
		var obj={x:arr[i][j][0],y:arr[i][j][1],v:parseFloat(scores[i][j])};//.toFixed(1)
		matrix.push(obj);
	}
}
	return matrix;//dict;
}
function getSimilarityValues(){
	var cloneCode=JSON.stringify(recommendationsList)
	$.ajax({
			type : 'POST',
			url : '/getSimilarityValues',
			data : {
				cloneCodeList:cloneCode
				},
			success: function (result) {
				scores=JSON.parse(result.scores)
				if(scores){
				scoreData=getDictionaryArray(scores);
				scatterlabels=recommendLabelsList;
				$("script[src='static/js/scatterplot.js']").remove();
				$('#similarityPlotView').children().remove();
				$('#similarityPlotView').append('<canvas id="chart-area2"></canvas>');
				
				$.ajax({
					  type: "GET",
					  url: "static/js/scatterplot.js",
					  dataType: "script"
					});
					$('#scatterPlotView').modal('toggle');
						  $('#appLoader').fadeOut();
						  toastr.success('Recommended Clone Methods have been successfully retrieved!');
				}

			},
			error: function (error) {
				console.log(error);
	        	  $('#appLoader').fadeOut();
			}
		});
	
}
function getCloneSeekerRecommendations() {	  
	var searchType=$("#searchType option:selected").text();
	var methodType=$("#methodType option:selected").text();

			$("#searchView .close").click();

		   $('#appLoader').fadeIn();		
		 $.ajax({
			type : 'POST',
			url : '/getCloneSeekerRecommendations',
			data : {
					query:$('#searchtext').val(),
					methodType:methodType,
					queryType:searchType,
					kvalue:$('#k-value').val()
				},
			success: function (result) {
				console.log(result);
				if(result.recommendations){			
						  	
						  identifierList=result.recommendedIdentifiers;
						  scoreList=result.scores;
						  recommendationsList=result.recommendations;
					      $('#recommendWindows').children().remove();
						  createRecommendWindows();
							  
						  for(var i=0;i<recomendEditorList.length;i++){
							recomendEditorList[i].setValue(result.recommendations[i]);  
							autoFormat(recomendEditorList[i]);
						  }
						 toastr.success('Recommended clone methods have been successfully retrieved!');

						 $('#appLoader').fadeOut();					  
				}

			},
			error: function (error) {
				console.log(error);
				$('#searchView').modal('toggle');
    			$('#appLoader').fadeOut();
				toastr["error"]("Something is wrong!");
			}
		});
	  
}

function searchText() {
    
	$('#recommend-code1').text('Search-Text1');
	$('#recommend-code2').text('Search-Text2');
	$('#recommend-code3').text('Search-Text3');
	$('#recommend-code4').text('Search-Text4');
	$('#recommend-code5').text('Search-Text5');
	$('#recommend-code6').text('Search-Text6');
	
}