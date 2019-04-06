import {ImageNet1000Class} from './imagenet1000.js';


let model;


const fetchData = async () => {
    model = await tf.loadLayersModel('./data/model/model.json');
    visualGroup.offLoadingModel();
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


const visualGroup = new Vue({
    el: '#visualGroup',
    data: {
        isLoadingModel: true,
        isSetingImg: false
    },
    methods: {
        offLoadingModel () {
            this.isLoadingModel = false;
        },
        onClassification () {
            this.isSetingImg = true;
        },
        setClassBar () {
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
        },
        setVisualisation () {
            const drawElement = document.getElementById('cnvs');
            const imageData = getImageData(drawElement);
            setGradImg(imageData, model);
        }
    }
});


const smoothGrad = new Vue({
    el: '#smoothGrad',
    methods: {
        applySmoothGrad (imageData, model, noizeLevel, sampleSize) {
            const noizeStd = 255*noizeLevel;
            const visualization = tf.tidy(() => {
                let gradImageData = tf.zeros([1, 224, 224]);
                let inputs = tf.browser.fromPixels(imageData, 3);
                inputs = tf.reverse(inputs, 2);  // RGB to BGR
                inputs = tf.cast(inputs, 'float32');
                const noize = tf.randomNormal([sampleSize, 224, 224, 3], 0.0, noizeStd, 'float32');
                inputs = noize.add(inputs);
                const predFunc = x => model.predict(x);
                const lossFunc = (x, y) => tf.losses.softmaxCrossEntropy(y, predFunc(x));
                const targets = predFunc(inputs).argMax(0);
                const labels = tf.oneHot(targets, 1000);
                const gradFunc = tf.grads(lossFunc);
                for (let i = 0; i < sampleSize; i++) {
                    const label = labels.slice([i, 0], [1, 1000]);
                    const input = inputs.slice([i, 0], [1, 224, 224, 3]);
                    const [gradient, _] = gradFunc([input, label]);
                    gradImageData = gradImageData.add(gradient.abs().max(3));
                }
                gradImageData = gradImageData.div(tf.scalar(sampleSize));
                return tf.squeeze(gradImageData);
            });
            return visualization;
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
                visualGroup.onClassification();
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


const setGradImg = async (imageData, model) => {
    const cnvs =  document.getElementById('cnvs-sg');
    cnvs.width = 224;
    cnvs.height = 224;
    let gradImgData = smoothGrad.applySmoothGrad(imageData, model, 0.2, 20);
    gradImgData = minMaxNormalization(gradImgData);
    await tf.browser.toPixels(gradImgData, cnvs);
};


const minMaxNormalization = (imageData) => {
    const dataMax = imageData.max();
    const dataMin = imageData.min();
    const normed = imageData.sub(dataMin).div(dataMax.sub(dataMin));
    return normed
}
