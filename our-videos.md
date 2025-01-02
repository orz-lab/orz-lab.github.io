---
layout: compress
title: Our Videos
---

<!DOCTYPE html>
<html>

<head>
    {% include head.html %}
</head>

<body>
    {% include navbar.html %}

    <!--Main Section-->
    <section class="hero is-fullheight" id="videos">
        <div class="hero-body">
            <div class="container has-text-centered">
                <div class="section">
                    <p class="subtitle is-uppercase has-text-weight-medium has-text-grey">Our Videos</p>
                </div>
                <div class="columns is-centered is-multiline is-mobile">
                    {% for video in site.data.our-videos %}
                    <div class="column has-text-centered is-paddingless is-marginless is-half-widescreen is-half-desktop is-half-fullhd is-half-tablet is-full-mobile"
                        id="video-card" style="margin-bottom: 15px;">
                        <div class="video-container" style="border: 2px dashed #ccc; padding: 10px;">
                            <iframe width="560" height="315" src="{{ video.link | replace: 'watch?v=', 'embed/' }}" frameborder="0" allowfullscreen></iframe>
                        </div>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </section>

    <!--Footer-->
    {% include footer.html %}
</body>

</html>
