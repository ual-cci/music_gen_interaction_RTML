$(document).ready(function(){

    $( "#sortable" ).sortable({
      placeholder: "ui-state-highlight"
    });

    $( "#sortable" ).disableSelection();



    $("div span").click(function(){
        $this = $(this).find(".image");
        $this.toggleClass("imageShow");
    });

    $("h1 span").click(function(){
        $(".image").toggleClass("imageShow");
    });

});
