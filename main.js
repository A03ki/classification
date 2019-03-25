import {ImageNet1000Class} from './imagenet1000.js';


let model;
tf.loadModel('./data/model/model.json')
    .then(pretrainedModel => {
        model = pretrainedModel;
    }).then(() => {
        classification.offLoadingModel();
    });


const userAgent = window.navigator.userAgent.toLowerCase();

if(userAgent.indexOf('msie') != -1 || userAgent.indexOf('trident') != -1) {
        alert('Internet Explorerは対応していません');
    } else if(userAgent.indexOf('edge') != -1) {
        alert('Edgeは対応していません');
    } else if(userAgent.indexOf('safari') != -1　&& userAgent.indexOf('chrome') === -1) {
        alert('Safariは対応していません');
    };


const classification = new Vue({
    el: '#classification',
    data: {
        isLoadingModel: true,
        isSetingImg: false
    },
    methods: {
        offLoadingModel() {
            this.isLoadingModel = false;
        },
        onClassification() {
            this.isSetingImg = true;
        },
        serClassBar() {
            const drawElement = document.getElementById('cnvs');
            const imageData = getImageData(drawElement);
            const accuracyScores = getAccuracyScores(imageData, model);
            const classRanking = putTopN(accuracyScores, 10);
            classBar.isGetingRnak = true;
            classRanking.forEach((val, idx) => {
                classBar.names[idx].message = ImageNet1000Class[val.index];
                classBar.items[idx].message = "0";
                classBar.styles[idx].width.width = "0%";
            });
            setTimeout(() => {
            classRanking.forEach((val, idx) => {
                classBar.items[idx].message = `${val.value * 100}`;
                classBar.styles[idx].width.width = `${val.value * 100}%`;
            });}, 1000);
        }
    }
});


const inputGroup = new Vue({
    el: '#inputGroup',
    data: {
        message: "画像を選択してください"
    },
    methods: {
        readFile (event) {
            const files = event.target.files;
            if(files.length > 0) {
                const file = files[0];
                this.message = files[0].name;
                const canvas = document.getElementById("cnvs");
                const ctx = canvas.getContext('2d');
                const image = new Image();
                const reader = new FileReader();
                reader.onload = (evt) => {
                    image.onload = () => {
                        const outputHeight = 224;
                        const outputWidth = image.naturalWidth * outputHeight / image.naturalHeight;
                        canvas.setAttribute("width", `${outputWidth}`);
                        canvas.setAttribute("height", `${outputHeight}`);
                        ctx.drawImage(image, 0, 0, outputWidth, outputHeight);
                    };
                    image.src = evt.target.result;
                };
                reader.readAsDataURL(file);
                classification.onClassification();
            }
        }
    }
});


const classBar = new Vue({
    el: '#classBar',
    data: {
        isGetingRnak: false,
        names: [
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        { message: "" },
        ],
        items: [
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        { message: "0" },
        ],
        styles: [
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } },
        { width: { "width" : "0%" } }
        ]
    }
});


const getImageData = (drawElement) => {  // resize
    const inputWidth = 224;
    const inputHeight = 224;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.setAttribute("width", `${inputWidth}`);
    canvas.setAttribute("height", `${inputHeight}`);
    ctx.drawImage(drawElement, 0, 0, inputWidth, inputHeight);
    let imageData = ctx.getImageData(0, 0, inputWidth, inputHeight);
    return imageData;
}


const getAccuracyScores = (imageData, model) => {
    const score = tf.tidy(() => {
        const channels = 3;
        let input = tf.fromPixels(imageData, channels);
        input = tf.reverse(input, 2);  // RGB to BGR
        input = tf.cast(input, 'float32');
        input = input.expandDims();
        return model.predict(input).dataSync();
    });
    return score;
}


const putTopN = (array, n) => {
    let hoge = [];
    array.forEach((val, idx) => {
        hoge.push({index : idx, value : val});
    });
    hoge.sort((a, b) => {
        return b.value - a.value;
    });
        return hoge.slice(0, n);
};
