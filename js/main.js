import {ImageNet1000Class} from './imagenet1000.js';


let model;
let mobilenet;
let resnet50 = undefined;


(async () => {
    mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    model = mobilenet;
    loading.offLoading();
})();


const userAgent = window.navigator.userAgent.toLowerCase();

if (userAgent.indexOf('msie') != -1 || userAgent.indexOf('trident') != -1) {
        alert('Internet Explorerは対応していません');
    } else if (userAgent.indexOf('edge') != -1) {
        alert('Edgeは対応していません');
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
            let input;
            if (model===mobilenet) {
                input  = applyPreprocessing(imageData, false);
                input = minMaxNormalization(input, [-1, 1]);
            } else {  // ResNet50
                input = applyPreprocessing(imageData);
            };
            const accuracyScores = model.predict(input).dataSync();
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
            loading.message = "visualisation: SmoothGrad"
            loading.onLoading();
            setTimeout(() => {
                writeGradImg(imageData, model);
                loading.offLoading();
            }, 100);
        }
    }
});


const applySmoothGrad = (imageData, model, noizeLevel, sampleSize) => {
    const visualization = tf.tidy(() => {
        let gradImageData = tf.zeros([1, 224, 224]);
        let inputs;
        if (model===mobilenet) {  // Mobilenet require RGB and scaling `-1~1`
            inputs = tf.randomNormal([sampleSize, 224, 224, 3], 0.0,  // noize
                                     1.0*noizeLevel, 'float32');
            inputs = inputs.add(minMaxNormalization(applyPreprocessing(imageData, false), [-1, 1]));
        } else {  // ResNet50
            inputs = tf.randomNormal([sampleSize, 224, 224, 3], 0.0,
                                     255*noizeLevel, 'float32');
            inputs = inputs.add(applyPreprocessing(imageData));
        };
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


const inputGroup = new Vue({
    el: '#inputGroup',
    template: `<div class="input-group mb-3">
                <label class="input-group-prepend">
                        <div class="btn btn-primary">
                            <i class="fas fa-folder-open" id="fileImage"></i>
                            Choose File
                            <input id="imegeForm" type="file" accept="image/jpeg,image/png" @change="readFile($event)">
                        </div>
                </label>
              <input type="text" class="form-control" readonly="" :value="message">
              </div>`,
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
    template: `<div v-cloak>
                <table class="table" v-show="isGetingRnak">
                    <thead>
                        <tr>
                            <th width="25%">Label</th>
                            <th width="75%">Probability</th>
                        </tr>
                    </thead>
                <tbody v-for="(style, index) in styles">
                    <tr>
                        <td>
                            {{ names[index].message }}
                        </td>
                        <td>
                            <div class="progress">
                                <div class="progress-bar" role="progressbar" v-bind:style="style.width">
                                    {{ items[index].message }}%
                                </div>
                            </div>
                        </td>
                    </tr>
                </tbody>
                </table>
               </div>`,
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


const applyPreprocessing = (imageData, rgb2bgr=true) => {
    const channels = 3;
    const output = tf.tidy(() => {
        let input = tf.browser.fromPixels(imageData, channels);
        if (rgb2bgr) input = tf.reverse(input, 2);  // RGB to BGR
        input = tf.cast(input, 'float32');
        input = input.expandDims();
        return input
    });
    return output;
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


const writeGradImg = async (imageData, model) => {
    const cnvs =  document.getElementById('cnvs-sg');
    cnvs.width = 224;
    cnvs.height = 224;
    let gradImgData = applySmoothGrad(imageData, model, 0.2, 20);
    gradImgData = minMaxNormalization(gradImgData);
    await tf.browser.toPixels(gradImgData, cnvs);
};


const minMaxNormalization = (imageData, range=[0, 1]) => {  // range[0]~range[1]にScaling
    const dataMax = imageData.max();
    const dataMin = imageData.min();
    let normed = imageData.sub(dataMin).div(dataMax.sub(dataMin));
    normed = normed.mul(tf.tensor(range[1] - range[0])).add(tf.tensor(range[0]));
    return normed
}


const loading = new Vue( {
    el: '#loading',
    template: `<div v-show="isLoading" class="loading">
                <img src="./data/loading.gif">
                <p>Now loading {{message}}</p>
            </div>`,
    data: {
        isLoading: true,
        message: "model: Mobilenet"
    },
    methods: {
        onLoading () {
            this.isLoading = true;
        },
        offLoading () {
            this.isLoading = false;
        },
    }
} );


const selectModel = new Vue( {
    el: '#selectModel',
    template: `<div class="btn-group">
                <button type="button" class="btn btn-primary dropdown-toggle" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
                    Model
                </button>
                <div class="dropdown-menu">
                    <a class="dropdown-item" v-bind:class="{ active: isActiveMobilenet, disabled : isDisabled }" v-on:click="setMobilenet">Mobilenet</a>
                    <a class="dropdown-item" v-bind:class="{ active: isActiveResNet50, disabled : isDisabled }" v-on:click="setResnet50">ResNet50</a>
                </div>
               </div>`,
    data: {
        isActiveMobilenet: true,
        isActiveResNet50: false,
        isDisabled: false
    },
    methods: {
        setMobilenet(){
            if (this.isActiveMobilenet===false) {
                this.swapCondition();
            };
            model = mobilenet;
        },
        setResnet50(){
            if (this.isActiveResNet50===false) {
                this.swapCondition();
            };
            if (resnet50 === undefined) {
                this.isDisabled = true;  // 何回も読み込ませないように選択できなくする
                loading.message = "model: ResNet50";
                loading.onLoading();
                (async () => {
                    resnet50 = await tf.loadLayersModel('./data/model/model.json');
                    model = resnet50;
                    this.isDisabled = false;
                    loading.offLoading();
                })();
            }
            model = resnet50;
        },
        swapCondition(){
            [this.isActiveMobilenet, this.isActiveResNet50] = [this.isActiveResNet50, this.isActiveMobilenet];
        }
    }
} );
