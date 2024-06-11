let im_width = 0;
let im_height = 0;
let resized_image = null;

document.addEventListener('DOMContentLoaded', function() {
    // Code to be executed after HTML document is loaded
    document.getElementById('fileInput').addEventListener('change', handleFile, false);
    document.getElementById('slider').addEventListener('change', changeK);
    document.getElementById('theme-toggle').addEventListener('click', changeTheme);
    document.getElementById('submit-btn').addEventListener('click', redraw);
});

function changeK() {
    // display the value of K when slider value is changed
    let slider = document.getElementById('slider');
    document.getElementById('slider-label').innerHTML = `Number of clusters: ${slider.value}`;
}

function changeTheme() {
    let body = document.body;
    let icon = document.getElementById('theme-toggle');
    body.dataset.bsTheme = body.dataset.bsTheme === "light" ? "dark" : "light";
    icon.innerHTML = icon.innerHTML === `<i class="bi bi-sun-fill"></i>` ? `<i class="bi bi-moon-stars-fill"></i>` : `<i class="bi bi-sun-fill"></i>`;
}

function handleFile(event) {
    var canvas = document.getElementById('canvas');
    var ctx = canvas.getContext('2d');
    var image = new Image();
    
    image.onload = function() {
        // make sure that the larger dimension is exactly 500px
        const max_dim = 500;
        var hRatio = Math.min(max_dim, window.innerWidth) / image.width;
        var vRatio = max_dim / image.height;
        var ratio  = Math.min( hRatio, vRatio );
        im_height = image.height*ratio;
        im_width = image.width*ratio;
        canvas.width = im_width;
        canvas.height = im_height;
        ctx.drawImage(image, 0, 0, image.width, image.height, 0, 0, image.width*ratio, image.height*ratio);
        resized_image = ctx.getImageData(0, 0, im_width, im_height);
    };

    var file = event.target.files[0];
    var reader = new FileReader();
    reader.onload = function(event) {
      image.src = event.target.result;
    };
    reader.readAsDataURL(file);
}

function redraw(){
    let button = document.getElementById('submit-btn');
    // disable button
    button.innerHTML = `Generate <div class="spinner-border spinner-border-sm" role="status"></div>`;
    button.classList.add('disabled');
    let colors = document.getElementById('color-list');
    var ctx = document.getElementById('canvas').getContext('2d');
    var imageData = resized_image;
    var pixels = imageData.data; // single list of R, G, B and A values of all the pixels
    var k = document.getElementById('slider').value;
    var result = clustering(pixels, k)
    console.log(result);

    // add the colors in the color list
    colors.innerHTML = `<li class="list-group-item list-group-item-secondary">Colors:</li>`;
    result.centroids.forEach(c => {
        let r = Math.round(c[0]);
        let g = Math.round(c[1]);
        let b = Math.round(c[2]);
        let hex = '#' + r.toString(16) + g.toString(16) + b.toString(16);
        let row = document.createElement('li');
        row.className = "list-group-item d-flex justify-content-between align-items-center";
        row.innerHTML = `<input type="color" class="form-control form-control-color me-2"value="${hex}"><code class="me-2">${hex}</code><code>rgb(${r}, ${g}, ${b})</code>`;
        colors.append(row);
    });

    // redraw image
    for(var i = 0; i < pixels.length; i += 4) {
      pixels[i] = result.centroids[result.clusters[i/4]][0]; // red channel
      pixels[i + 1] = result.centroids[result.clusters[i/4]][1]; // green channel
      pixels[i + 2] = result.centroids[result.clusters[i/4]][2]; // blue channel
    }
    ctx.putImageData(imageData, 0, 0);
    // disable button
    button.innerHTML = `Generate <i class="bi bi-file-earmark-image"></i>`;
    button.classList.remove('disabled');
}


//---------------------------------------------------------------------------------------------------------------
// K means clustering
// modified from code at: https://medium.com/geekculture/implementing-k-means-clustering-from-scratch-in-javascript-13d71fbcb31e

const MAX_ITERATIONS = 50;
const channels = 3;

function calcMeanCentroid(dataSet, start, end) {
  const n = end - start;
  let mean = [];
  for (let i = 0; i < channels; i++) {
    mean.push(0);
  }
  for (let i = start; i < end; i++) {
    for (let j = 0; j < channels; j++) {
      mean[j] = mean[j] + dataSet[i][j] / n;
    }
  }
  return mean;
}

function getRandomCentroids(dataset, k) {
  // implementation of a variation of naive sharding centroid initialization method
  // (not using sums or sorting, just dividing into k shards and calc mean)
  // https://www.kdnuggets.com/2017/03/naive-sharding-centroid-initialization-method.html
  const numSamples = dataset.length;
  // Divide dataset into k shards:
  const step = Math.floor(numSamples / k);
  const centroids = [];
  for (let i = 0; i < k; i++) {
    const start = step * i;
    let end = step * (i + 1);
    if (i + 1 === k) {
      end = numSamples;
    }
    centroids.push(calcMeanCentroid(dataset, start, end));
  }
  return centroids;
}

function compareCentroids(a, b) {
  for (let i = 0; i < channels; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

function shouldStop(oldCentroids, centroids, iterations) {
  if (iterations > MAX_ITERATIONS) {
    return true;
  }
  if (!oldCentroids || !oldCentroids.length) {
    return false;
  }
  let sameCount = true;
  for (let i = 0; i < centroids.length; i++) {
    if (!compareCentroids(centroids[i], oldCentroids[i])) {
      sameCount = false;
    }
  }
  return sameCount;
}

// Calculate Squared Euclidean Distance
function getDistanceSQ(a, b) {
  const diffs = [];
  for (let i = 0; i < channels; i++) {
    diffs.push(a[i] - b[i]);
  }
  return diffs.reduce((r, e) => (r + (e * e)), 0);
}

// Returns a label for each piece of data in the dataset. 
function getLabels(dataSet, centroids) {
  // prep data structure:
  const labels = {};
  for (let c = 0; c < centroids.length; c++) {
    labels[c] = {
      points: [],
      centroid: centroids[c],
    };
  }
  // For each element in the dataset, choose the closest centroid. 
  // Make that centroid the element's label.
  for (let i = 0; i < dataSet.length; i++) {
    const a = dataSet[i];
    let closestCentroid, closestCentroidIndex, prevDistance;
    for (let j = 0; j < centroids.length; j++) {
      let centroid = centroids[j];
      if (j === 0) {
        closestCentroid = centroid;
        closestCentroidIndex = j;
        prevDistance = getDistanceSQ(a, closestCentroid);
      } else {
        // get distance:
        const distance = getDistanceSQ(a, centroid);
        if (distance < prevDistance) {
          prevDistance = distance;
          closestCentroid = centroid;
          closestCentroidIndex = j;
        }
      }
    }
    // add point to centroid labels:
    labels[closestCentroidIndex].points.push(a);
  }
  return labels;
}

function getPointsMean(pointList) {
  const totalPoints = pointList.length;
  const means = [];
  for (let j = 0; j < channels; j++) {
    means.push(0);
  }
  for (let i = 0; i < pointList.length; i++) {
    const point = pointList[i];
    for (let j = 0; j < channels; j++) {
      const val = point[j];
      means[j] = means[j] + val / totalPoints;
    }
  }
  return means;
}

function recalculateCentroids(dataSet, labels, k) {
  // Each centroid is the geometric mean of the points that
  // have that centroid's label. Important: If a centroid is empty (no points have
  // that centroid's label) you should randomly re-initialize it.
  let newCentroid;
  const newCentroidList = [];
  for (const k in labels) {
    const centroidGroup = labels[k];
    if (centroidGroup.points.length > 0) {
      // find mean:
      newCentroid = getPointsMean(centroidGroup.points);
    } else {
      // get new random centroid
      newCentroid = getRandomCentroids(dataSet, 1)[0];
    }
    newCentroidList.push(newCentroid);
  }
  return newCentroidList;
}

function clustering(pixels, k) {
    // Convert input pixels to suitable form
    let dataset = [];
    for (let i = 0; i < pixels.length; i+=4) {
        dataset.push([pixels[i], pixels[i+1], pixels[i+2], i/4]);
    }
    
    // Initialize book keeping variables
    let iterations = 0;
    let oldCentroids, labels, centroids;

    // Initialize centroids randomly
    centroids = getRandomCentroids(dataset, k);
    console.log(centroids);

    // Run the main k-means algorithm
    while (!shouldStop(oldCentroids, centroids, iterations)) {
        // Save old centroids for convergence test.
        oldCentroids = [...centroids];
        iterations++;

        // Assign labels to each datapoint based on centroids
        labels = getLabels(dataset, centroids);
        centroids = recalculateCentroids(dataset, labels, k);
    }

    let clusters = [];
    for (let i = 0; i < dataset.length; i++) {
      clusters.push(0);
    }
    for (let i = 1; i<k; i++) {
      let points = labels[i].points;
      for (let j = 0; j<points.length; j++) {
        clusters[points[j][3]] = i;
      }
    }
    const results = {
        clusters: clusters,
        centroids: centroids,
        iterations: iterations,
        converged: iterations <= MAX_ITERATIONS,
    };
    return results;
}