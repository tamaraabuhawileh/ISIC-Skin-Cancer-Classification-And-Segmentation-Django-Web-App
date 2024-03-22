  function readURL(input) {
    if (input.files && input.files[0]) {
      var reader = new FileReader();
 
      reader.onload = function (e) {
        var uploadedImg = document.getElementById('uploadedImg');
        if (uploadedImg) {
          uploadedImg.style.backgroundImage = 'url(' + e.target.result + ')';
        }
      };
 
      reader.readAsDataURL(input.files[0]);
    }
  }
 
  window.onload = function () {
    var form = document.getElementById('imageUploadForm');
    var fileInput = document.getElementById('file');
    var uploadedImg = document.getElementById('uploadedImg');
    var helpText = document.getElementById('helpText');
 
    if (fileInput && form && uploadedImg && helpText) {
      fileInput.addEventListener('change', function () {
        // Check if the form is not already in a loading or loaded state
        if (!form.classList.contains('loading') && !form.classList.contains('loaded')) {
          readURL(this);
          form.classList.add('loading');
        }
      });
 
      uploadedImg.addEventListener('animationend', function () {
        form.classList.add('loaded');
      });
 
    }
  };
 
  function uploadImage(input) {
    var form = document.getElementById('imageUploadForm');
    // Check if the form is not already in a loading or loaded state
    if (!form.classList.contains('loading') && !form.classList.contains('loaded')) {
      if (input.files && input.files[0]) {
        var reader = new FileReader();
 
        reader.onload = function (e) {
          var uploadedImg = document.getElementById('uploadedImg');
          if (uploadedImg) {
            uploadedImg.style.backgroundImage = 'url(' + e.target.result + ')';
            form.submit(); // Submit the form once the file is picked and image is displayed
          }
        };
 
        reader.readAsDataURL(input.files[0]);
      }
    }
  }