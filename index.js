const mvae = require('@magenta/music/node/music_vae');
const core = require('@magenta/music/node/core');
const fs = require('fs');
const path = require('path');
const {midiFilesCatchy, midiFilesDark, midiFilesEDM, midiFilesEmotional, midiFilesPop, midiFilesRnB, midiFilesCatchy16Bar, midiFilesDark16Bar, midiFilesEDM16Bar, midiFilesEmotional16Bar, midiFilesPop16Bar} = require('./midi.js');

// modelCheckpoint = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_2bar_small';
modelCheckpoint = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_med_q2';
// Trained with a strong prior (low KL divergence), which is better for sampling.
// modelCheckpoint = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_4bar_med_lokl_q2';
// const modelCheckpoint = 'https://storage.googleapis.com/magentadata/js/checkpoints/music_vae/mel_16bar_small_q2';
const MEL_BARS = 4;

const model = new mvae.MusicVAE(modelCheckpoint);
let i = 1;
const embeddingsFileName = 'embeddings_mel_4bar_med_q2_NEW.tsv';
const metaDataFileName = 'metadata_mel_4bar_med_q2_NEW.tsv';

init().then(() => console.log('Finished!'));

async function init() {
    console.log('Starting...');
    await model.initialize();
    console.log('Creating Embedding Files...');
    await createEmbeddingFileFor('catchy', midiFilesCatchy);
    await createEmbeddingFileFor('dark', midiFilesDark);
    await createEmbeddingFileFor('edm', midiFilesEDM);
    await createEmbeddingFileFor('emotional', midiFilesEmotional);
    await createEmbeddingFileFor('pop', midiFilesPop);
    await createEmbeddingFileFor('rnb', midiFilesRnB);
    // For 16bar long only sequences
    // await createEmbeddingFileFor('catchy', midiFilesCatchy16Bar);
    // await createEmbeddingFileFor('dark', midiFilesDark16Bar);
    // await createEmbeddingFileFor('edm', midiFilesEDM16Bar);
    // await createEmbeddingFileFor('emotional', midiFilesEmotional16Bar);
    // await createEmbeddingFileFor('pop', midiFilesPop16Bar);
    console.log('Joining all files...')
    joinEmbeddingFiles();
}

async function createEmbeddingFileFor(category, midiFilesUrl) {
    const melodies = [];
    for (const midi of midiFilesUrl) {
        const file = fs.readFileSync(__dirname + midi);
        // console.log(file);
        const sequence = await core.midiToSequenceProto(file);
        melodies.push(sequence);
    }

    // 1. Encode the input into MusicVAE, get back a z.
    const quantizedMels = [];
    melodies.forEach((m) => quantizedMels.push(core.sequences.quantizeNoteSequence(m, 4)));

    // 1b. Split this sequence into 2 bar chunks.
    let chunks = [];
    const metaDataNames = [];
    quantizedMels.forEach((m) => {
        // if you want to split the sequence into 2 bar chunks,
        // then if the sequence has 16th note quantization,
        // that will be 32 steps for each 2 bars (so a chunkSize of 32)
        const length = 16 * MEL_BARS; // = 32
        const melChunks = core.sequences.split(core.sequences.clone(m), length);
        chunks = chunks.concat(melChunks); // Array of 2 bar chunks

        // Prepare metaData
        for (let j = 1; j <= melChunks.length; j++) {
            metaDataNames.push(j.toString());
        }
    });

    // Prepare metaData
    const metaDataChunkNames = [];
    let x = 0;
    let currentFileName = '';
    for (let j = 0; j < metaDataNames.length; j++) {
        if (metaDataNames[j] !== '1') {
            currentFileName = path.basename(midiFilesUrl[x-1], '.mid');
            metaDataChunkNames.push(currentFileName + '-' + metaDataNames[j]);
            continue
        }
        x++;
        currentFileName = path.basename(midiFilesUrl[x-1], '.mid');
        metaDataChunkNames.push(currentFileName + '-' + metaDataNames[j]);
    }

    // Get for every chunk a MusicVAE z value
    const z = await model.encode(chunks);  // shape of z is [chunks, 256]
    const attributeVectorZMean = z.mean(0, true); // mean of all chunks. Shape: [1, 256]

    // z.print(true);
    // attributeVectorZMean.print(true);
    // console.log(JSON.stringify(attributeVectorZMean.arraySync()));

    // Preparing the data for csv writing
    const separator = '\t';
    let csvContent = '';
    // let csvMetaData = 'id' + separator + 'category' + separator + 'name' + '\r\n';
    let csvMetaData = '';
    z.arraySync().forEach(function(rowArray, index) {
        let row = rowArray.join(separator);
        csvContent += row + "\r\n";
        csvMetaData += i + separator + category + separator + metaDataChunkNames[index] + "\r\n";
        i++;
    });
    attributeVectorZMean.arraySync().forEach(function(rowArray, index) {
        let row = rowArray.join(separator);
        csvContent += row + "\r\n";
        csvMetaData += i + separator + category + '_mean' + separator + '0' + "\r\n";
        i++;
    });

    // Write embeddings & metadata in CSV-files
    try {
        if (!fs.existsSync('./embeddings') && !fs.existsSync('./metadata')){
            fs.mkdirSync('./embeddings');
            fs.mkdirSync('./metadata');
        }
        fs.writeFileSync('./embeddings/embedding_' + category + '.tsv', csvContent, {flag: 'w'})
        console.log('File embedding_' + category + '.tsv' + ' successfully created.')
        fs.writeFileSync('./metadata/embedding_' + category + '_metadata.tsv', csvMetaData, {flag: 'w'})
        console.log('File embedding_' + category + '_metadata.tsv' + ' successfully created.')
        //file written successfully
    } catch (err) {
        console.error(err)
    }
}

function joinEmbeddingFiles() {
    // Deletes final files if they already exist
    if (fs.existsSync('./' + embeddingsFileName)) {
        //file exists
        fs.rmSync('./' + embeddingsFileName);
    }
    if (fs.existsSync('./' + metaDataFileName)) {
        //file exists
        fs.rmSync('./' + metaDataFileName);
    }

    fs.readdirSync('./embeddings').forEach((file) => { if(fs.lstatSync('./embeddings/' + file).isFile()) fs.appendFileSync('./' + embeddingsFileName, fs.readFileSync('./embeddings/' + file).toString()) });
    console.log('File ' + embeddingsFileName + ' successfully created.');
    fs.readdirSync('./metadata').forEach((file) => { if(fs.lstatSync('./metadata/' + file).isFile()) fs.appendFileSync('./' + metaDataFileName, fs.readFileSync('./metadata/' + file).toString()) });
    console.log('File ' + metaDataFileName + ' successfully created.');
    console.log('Deleting Temp files...');
    fs.rmdirSync('./embeddings', { recursive: true });
    fs.rmdirSync('./metadata', { recursive: true });
}

