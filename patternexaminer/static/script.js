
var positiveClass = 'success';
var negativeClass = 'danger';
var selectedExperimentClass = 'info';
var currentMousePos = {x: -1, y: -1};
var latestExperimentsHash;
var selectedExperiment = -2;
var selectedLabel = -2;
var pendingExperiments = 1;
var sampleAllSelected = false;


$(document).ready(function() {

  $(document).mousemove(function(event) {
      currentMousePos.x = event.pageX;
      currentMousePos.y = event.pageY;
  });

  $(document).keypress(function(event) {
    
    var keycode = (event.keyCode ? event.keyCode : event.which);
    var element = document.elementFromPoint(currentMousePos.x, currentMousePos.y);
    // console.log(keycode)
    // 1
    if (keycode == '49'){
      $('select[name=input_type]').val(1).change();
    }
    // 2
    if (keycode == '50'){
      $('select[name=input_type]').val(2).change();
    }
    // 3
    if (keycode == '51'){
      $('select[name=input_type]').val(3).change();
    }
    // // a
    // if (keycode == '97'){
    //   setPositive(element)
    // }
    // // s
    // if (keycode == '115'){
    //   setNegative(element)
    // }
    
  });

  if ($('select[name=regex_name]').children().length != 0) {
    $('select[name=regex_name]').change(function() {
      $('select[name=regex_pattern]').children().hide();
      var children = $('option[name_id=' + this.value + ']');
      children.show();
      // Selecting the pattern that corresponds to the name.
      children[0].selected = 'selected'
    });

    $('select[name=regex_name]').change();
  }

  update_latest_experiments();
  setInterval(update_latest_experiments, 10000, -1);

  $('#latest_experiments').on('click', 'tr', function(event) {
    var row = $(event.currentTarget);
    id = row.attr('experiment_id');
    selectedExperiment = id;
    $('#latest_experiments').find('tr').removeClass(selectedExperimentClass)
    row.addClass(selectedExperimentClass);
    clearAllResults();
    $('#labelsCounts').load('clusters_sizes?', $.param({experiment_id: id}));
    $('#statistics').load('get_statistics?', $.param({experiment_id: id}));

  });

  $('#labelsCounts').on('click', 'a', function(event) {
    $('#labelsCounts a').removeClass('active');
    $(this).addClass('active');
    selectedLabel = event.currentTarget.getAttribute('cluster_label');
    getSample();
  });

  $('#clusterBtn').click(function() {
    var clusterData = $('#clusterForm').serializeArray();
    clusterData.push({name: 'parent_id', value: selectedExperiment});
    clusterData.push({name: 'parent_label', value: selectedLabel});
    var inputType = $('select[name=input_type] option:selected').text();
    if (inputType == 'Cluster' && (selectedExperiment == -2 || selectedLabel == -2)) {
      alert('Cluster label not selected');
      return false;
    }
    pendingExperiments += 1;
    $('#experimentLoading').show();
    console.log(pendingExperiments)
    $.post('cluster', clusterData, function(response) {
      if (response.success === 1) {
        setIntervalX(update_latest_experiments, 1000, 10);
        // TODO: Tweak this interval time according to actual usage. 
        setIntervalX(update_latest_experiments, 3000, 10);
      }
    })
  });

  $('#clusterSample').on('mousedown', 'tr', function(event) {

    var resultId = this.getAttribute('result_id'); 
    var evaluation;
    // Left click, positive row.
    if (event.button === 0) {
      // colorRowPositive(this)
      evaluation = true;
    }
    // Right click, negative row.
    if (event.button === 2) {
      // colorRowNegative(this)
      evaluation = false;
    }

    setEvaluation(resultId, evaluation);

  });

  $('select[name=input_type]').change(function() {
    if (this.selectedOptions[0].text == 'Extractor') {
      $('select[name=regex_name]').closest('.form-group').css('visibility', 'visible');
      $('select[name=regex_pattern]').closest('.form-group').css('visibility', 'visible');
    } else {
      $('select[name=regex_name]').closest('.form-group').css('visibility', 'hidden');
      $('select[name=regex_pattern]').closest('.form-group').css('visibility', 'hidden');
    }
  });

  // $('select[name=input_type]').change();

  // Give the filtering and/or checkboxes a bit of a radio button feel, for comfort. 
  $("input[name=filteringType").click(function() {
    var group = "input:checkbox[name='"+$(this).prop("name")+"']";
    var checked = !$(this).prop("checked");
    $(group).prop("checked",false);
    if (!checked) {
      $(this).prop("checked",true);
      $('#filterBoxes').slideDown();
    } else {
      $('#filterBoxes').slideUp()
    }
  });

  $("input[name=sample_all").click(function() {
    sampleAllSelected = $(this).prop("checked");
  });

  $("input[name=sample-observed").click(function() {
    if (!sampleAllSelected) {
      var checked = $(this).prop("checked");
      $("input[name=sample_all").prop("checked",checked);
    }
  });

  $(".filter-input").keyup(function (e) {
    // The enter key.
    if (e.keyCode == 13) {
        getSample();
    }
  });

  $('#evaluatePositive').click(function() {
    evaluateAll(true);
  });

  $('#evaluateNegative').click(function() {
    evaluateAll(false);
  });

});


function allEvaluated() {
  var selectorBase = '#clusterSample table tr';
  var selectorPrefix = selectorBase + '.';
  var positiveCount = $(selectorPrefix + positiveClass).length;
  var negativeCount = $(selectorPrefix + negativeClass).length;
  var totalCount = $(selectorBase).length;
  if (positiveCount + negativeCount == totalCount) {
    return true;
  } else {
    return false; 
  }
}


function evaluateAll(evaluation) {
  var rowChoice = null;
  if (allEvaluated()) {
    rowChoice = evaluation;
  } 
  $('#clusterSample table tr').each(function () {
    if (getRowEvaluation($(this)) == rowChoice) {
      setEvaluation(this.getAttribute('result_id'), evaluation);
    }
  });
}


function setEvaluation(resultId, evaluation) {
  // Color the table row.
  var row = $('tr[result_id=' + resultId + ']');
  if (evaluation == true) {
    row.removeClass(negativeClass);
    row.toggleClass(positiveClass);
  } else {
    row.removeClass(positiveClass);
    row.toggleClass(negativeClass);
  }
  evaluation = getRowEvaluation(row);
  // Set evaluation in database.
  $.post(
  'set_evaluation', 
  {result_id: resultId, evaluation: evaluation, experiment_id: selectedExperiment},
  function(result) {
    if (result.success !== 1) {
      alert('server error')
    }
  })
  .fail(function() {
    alert('server error')
  });
}


function getSample() {

    var sampleType = $('input[name=sampleOption]:checked').val();
    if (sampleType == 'random') {
      var sampleUrl = 'get_random_sample?';
    }
    if (sampleType == 'heterogenous') {
      var sampleUrl = 'get_heterogenous_sample?';
    }

    samplingParameters = $('#sampling-form').serializeArray();
    samplingParameters.push({name: 'experiment_id', value: selectedExperiment});
    samplingParameters.push({name: 'label', value: selectedLabel});

    $.getJSON(sampleUrl, samplingParameters, function(result) {
      showEvaluationButtons();
      $('#clusterSample').html(result.html)
      if (result.filteredSize != 0) {
        $('#filteredStats').html(' | Filtered: ' + result.filteredSize)
      } else {
        $('#filteredStats').html('');
      }
    });

    getStatistics(id, selectedLabel);
    clearStatistics();
    clearSample();
}


function clearAllResults() {
  clearClusterLabels();
  clearStatistics()
  clearSample();
  hideEvaluationButtons();
}


function showEvaluationButtons() {
  if ($('.groupButtons').css('visibility') == 'hidden') {
    $('.groupButtons').css('visibility', 'visible');
  }
}


function hideEvaluationButtons() {
  $('.groupButtons').css('visibility', 'hidden');
}


function clearSample() {
  $('#clusterSample table').css('visibility', 'hidden');
}


function clearClusterLabels() {
  // $('#labelsCounts').css('visibility', 'hidden');
  $('#labelsCounts').html('');
  selectedLabel = -2;
}


function clearStatistics() {
  $('#positiveStats, #negativeStats, #observedStats').html('');
}


function getStatistics(experiment_id, label) {
  $('#statistics').load('get_statistics?', $.param({experiment_id: experiment_id, label: label}));
}


function update_latest_experiments() {
  $.getJSON('latest_experiments').done(function(response) {
    if (latestExperimentsHash !== response.latestHash) {
      removePending();
      $('#latest_experiments').html(response.html);
      $('[experiment_id=' + selectedExperiment + ']').addClass(selectedExperimentClass);
      latestExperimentsHash = response.latestHash;
      $('[data-toggle="tooltip"]').tooltip();
      $('[data-toggle="popover"]').popover({container: 'body'});
    } 
  });
}


function removePending() {
  if (pendingExperiments != 0) {
    pendingExperiments -= 1;
    if (pendingExperiments == 0) {
      $('#experimentLoading').hide();
    }
  }
}


function setIntervalX(callback, delay, repetitions) {
  var x = 0;
  var intervalID = window.setInterval(function () {
    callback();
    if (++x === repetitions) {
      window.clearInterval(intervalID);
    }
  }, delay);
}


function getRowEvaluation(row) {
  if (row.hasClass(positiveClass)) {
    return 1;
  }
  if (row.hasClass(negativeClass)) {
    return 0;
  }
  return null;
}


function colorRowPositive(resultId) {
  var row = $('tr[result_id=' + result_id + ']');
  row.removeClass(negativeClass);
  row.toggleClass(positiveClass);
}


function colorRowNegative(resultId) {
  var row = $('tr[result_id=' + result_id + ']');
  row.removeClass(positiveClass);
  row.toggleClass(negativeClass);
}

