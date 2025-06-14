// Projects Slideshow Functionality
let slideIndex = 1;

// When the document is loaded, initialize the slideshow
document.addEventListener('DOMContentLoaded', function() {
  showSlides(slideIndex);
});

// Next/previous controls
function plusSlides(n) {
  showSlides(slideIndex += n);
}

// Thumbnail image controls
function currentSlide(n) {
  showSlides(slideIndex = n);
}

function showSlides(n) {
  let i;
  let slides = document.getElementsByClassName("slide");
  let dots = document.getElementsByClassName("dot");
  
  if (slides.length === 0) return; // No slides to show
  
  if (n > slides.length) {slideIndex = 1}
  if (n < 1) {slideIndex = slides.length}
  
  for (i = 0; i < slides.length; i++) {
    slides[i].style.display = "none";
  }
  
  for (i = 0; i < dots.length; i++) {
    dots[i].className = dots[i].className.replace(" active", "");
  }
  
  slides[slideIndex-1].style.display = "block";
  dots[slideIndex-1].className += " active";
}

// Auto slideshow functionality
let slideInterval = setInterval(function() {
  plusSlides(1);
}, 10000); // Change slide every 10 seconds

// Pause slideshow when mouse hovers over it
document.querySelector('.projects-slideshow-container').addEventListener('mouseenter', function() {
  clearInterval(slideInterval);
});

// Resume slideshow when mouse leaves
document.querySelector('.projects-slideshow-container').addEventListener('mouseleave', function() {
  slideInterval = setInterval(function() {
    plusSlides(1);
  }, 10000);
});