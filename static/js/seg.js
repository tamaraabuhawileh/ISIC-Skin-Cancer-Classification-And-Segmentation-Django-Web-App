document.getElementById('photoInput').addEventListener('change', function(event) {
  const file = event.target.files[0];
  const image = document.getElementById('imagePreview');
  
  if (file) {
    const reader = new FileReader();
    
    reader.onload = function(e) {
      image.src = e.target.result;
      image.style.display = 'block';
    }
    
    reader.readAsDataURL(file);
  } else {
    image.style.display = 'none';
  }
});

document.getElementById('uploadForm').addEventListener('submit', function(event) {
  event.preventDefault();
  // Add your upload logic here
  // For demonstration purposes, you can use JavaScript Fetch or Ajax to send the image data to a server
  // Example: 
  // const formData = new FormData(this);
  // fetch('your_upload_endpoint', {
  //   method: 'POST',
  //   body: formData
  // })
  // .then(response => {
  //   // Handle response
  // })
  // .catch(error => {
  //   // Handle error
  // });
});
