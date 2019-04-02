import {ImageNet1000Class} from './imagenet1000.js';

let model;

const fetchData = async () => {
    model = await tf.loadLayersModel('./data/model/model.json');
    classification.offLoadingModel();
};

fetchData();


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
            const SmoothGrad = applySmoothGrad(imageData, model, 0.2, 100);
            console.log(SmoothGrad);
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
        let input = tf.browser.fromPixels(imageData, channels);
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


const applySmoothGrad = (imageData, model, noizeLevel, sampleSize) => {
    const gradList = [];
    const channels = 3;
    const visualization = tf.tidy(() => {
        let input = tf.fromPixels(imageData, channels);
        input = tf.reverse(input, 2);  // RGB to BGR
        input = tf.cast(input, 'float32');
        console.log(model)
        for (let i = 0; i < sampleSize; i++) {
            const noize = tf.randomNormal([224, 224, 3], 0, 255*noizeLevel, 'float32');
            let inputNoize = input.add(noize);
            inputNoize = tf.variable(inputNoize);
            console.log(inputNoize);
            const f = x => model.predict(x);
            const g = tf.grad(f);
            g(inputNoize.expandDims()).print();
            const output = model.predict(input.expandDims());
            console.log(output);
            const label = tf.oneHot(tf.tensor1d([output.argMax()], 'float32'), 1000);
            const loss = tf.losses.softmaxCrossEntropy(label, output)
            //const g = tf.grad(loss);
            gradList.push(g(inputNoize));
        }
        gradImageData = tf.squeeze(tf.tensor1d(gradList).abs().max(2).mean(0));
        return gradImageData.dataSync();
    });
    return visualization;
}

const tfVisualization = () => {
    const ImageData = gradImageData();
    tf.toPixels(ImageData, cnvs);
}
