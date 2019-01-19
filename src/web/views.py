from django.shortcuts import render

# Create your views here.
html = open('src/web/template/imageModal.html', 'r', encoding="utf-8")
html_modal = html.read()

html = open('src/web/template/imageItem.html', 'r', encoding="utf-8")
html_item = html.read()

html = open('src/web/template/showImage.html', 'r', encoding="utf-8")
html_js = html.read()
