// We do NOT adjust this code as requested; we simply apply it to our new structure
gsap.set(".scaleDown", { xPercent: -50, yPercent: -50 });

// Animate from scale=1 (or bigger if you'd like) down to 0.6667
// pinned to .container
gsap.to(".scaleDown", {
  scale: 0.5,
  scrollTrigger: {
    trigger: ".container",
    pin: ".container",
    scrub: true
  }
});

