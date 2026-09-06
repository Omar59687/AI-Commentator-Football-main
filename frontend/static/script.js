async function upload() {

    const status = document.getElementById("status");
    const spinner = document.getElementById("spinner");

    const file = document
        .getElementById("video")
        .files[0];

    if (!file) {
        status.className = "alert alert-danger";
        status.innerText = "Please choose a video.";
        return;
    }

    status.className = "alert alert-info";
    status.innerText = "Uploading video...";

    spinner.style.display = "block";

    const data = new FormData();
    data.append("file", file);

    status.innerText = "Generating AI commentary...";

    const response = await fetch("/generate", {
        method: "POST",
        body: data
    });

    if (!response.ok) {
        spinner.style.display = "none";
        status.className = "alert alert-danger";
        status.innerText = "Something went wrong.";
        return;
    }

    const result = await response.json();

    spinner.style.display = "none";

    status.className = "alert alert-success";

    status.innerHTML = `
    <h5>English Commentary</h5>
    <p id="english-commentary"></p>
    <video class="commentary-video" controls></video>

    <hr>

    <h5>Arabic Commentary</h5>
    <p id="arabic-commentary" dir="rtl"></p>
    <video class="commentary-video" controls></video>
    `;

    document.getElementById("english-commentary").textContent = result.english;
    document.getElementById("arabic-commentary").textContent = result.arabic;

    const videos = status.querySelectorAll("video");
    const cacheBuster = Date.now();
    videos[0].src = `${result.english_video_url}?v=${cacheBuster}`;
    videos[1].src = `${result.arabic_video_url}?v=${cacheBuster}`;
}
