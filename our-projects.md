---
layout: compress
title: Our Projects
---


<!DOCTYPE html>
<html>

<head>
    {% include head.html %}
</head>

<body>
    {% include navbar.html %}

    <!--Main Section-->
    <section class="hero is-fullheight" id="project">
        <div class="hero-body">
            <div class="container has-text-centered">
                <div class="section">
                    <p class="subtitle is-uppercase has-text-weight-medium has-text-grey">Projects</p>
                </div>
                <div class="columns is-centered is-multiline is-mobile">
                    {% for project in site.data.our-projects %}
                    <div class="column has-text-centered is-paddingless is-marginless is-one-third-widescreen is-one-third-desktop is-one-fifth-fullhd is-one-third-tablet is-two-fifths-mobile is-three-quarters-touch"
                        id="project-card">
                        <a href="{{project.link}}" target="_blank">
                            <div class="has-background-black card">
                                <figure class="image is-3by1" style="background-image: url({{project.image}});">
                                </figure>
                                <div class="card-content">
                                    <h1 class="title has-text-white is-size-4">{{ project.name }}</h1>
                                    <p class="has-text-white has-text-weight-light content">{{ project.description |
                                        truncate: 80}}</p>
                                </div>
                            </div>
                        </a>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </section>
    </div>

    <!--Footer-->
    {% include footer.html %}
</body>

</html>