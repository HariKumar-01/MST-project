document.getElementById("languageSelector").addEventListener("change", function () {
    const language = this.value;
    const articles = document.querySelectorAll(".article-card");

    articles.forEach(article => {
        const heading = article.querySelector("h3");
        heading.textContent = article.getAttribute(`data-${language.toLowerCase()}`);
    });
});
