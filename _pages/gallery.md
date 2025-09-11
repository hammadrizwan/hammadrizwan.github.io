---
layout: page
title: Gallery
permalink: /gallery/
toc: false
nav: false       # 👈 this makes it appear in the navbar
nav_order: 7   # 👈 controls position (lower numbers come earlier)
---

<div class="container px-0">
  <div class="row row-cols-1 row-cols-sm-2 row-cols-md-1 g-1">
    {%- for it in site.data.gallery -%}
      <div class="col">
        {% include figure.liquid
           path=it.path
           alt=it.alt
           caption=it.caption
           loading="lazy"
           class="w-100 rounded"
        %}
      </div>
    {%- endfor -%}
  </div>
</div>
