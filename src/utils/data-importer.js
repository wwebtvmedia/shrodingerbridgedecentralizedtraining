class DataImporter {
  constructor() {
    this.supportedImageFormats = [
      "image/jpeg",
      "image/png",
      "image/webp",
      "image/gif",
    ];
    this.supportedTextFormats = ["text/plain", "application/json", "text/csv"];
    this.maxImageSize = 1024 * 1024 * 10; // 10MB
    this.maxTextSize = 1024 * 1024 * 5; // 5MB
  }

  async importImages(files) {
    const images = [];

    for (const file of files) {
      if (!this.supportedImageFormats.includes(file.type)) {
        console.warn(`Unsupported image format: ${file.type}`);
        continue;
      }

      if (file.size > this.maxImageSize) {
        console.warn(`Image too large: ${file.name} (${file.size} bytes)`);
        continue;
      }

      try {
        const imageData = await this.processImage(file);
        images.push(imageData);
      } catch (error) {
        console.error(`Failed to process image ${file.name}:`, error);
      }
    }

    return images;
  }

  async processImage(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        const img = new Image();

        img.onload = () => {
          // Resize image to standard training size
          const canvas = document.createElement("canvas");
          canvas.width = 96; // Standard training size
          canvas.height = 96;

          const ctx = canvas.getContext("2d");

          // Calculate scaling to maintain aspect ratio
          const scale = Math.min(
            canvas.width / img.width,
            canvas.height / img.height,
          );

          const width = img.width * scale;
          const height = img.height * scale;
          const x = (canvas.width - width) / 2;
          const y = (canvas.height - height) / 2;

          // Draw resized image
          ctx.fillStyle = "#000000";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, x, y, width, height);

          // Convert to tensor format (simulated for prototype)
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

          resolve({
            name: file.name,
            type: file.type,
            size: file.size,
            width: canvas.width,
            height: canvas.height,
            data: imageData,
            tensor: this.imageDataToTensor(imageData),
            timestamp: Date.now(),
            url: event.target.result,
          });
        };

        img.onerror = reject;
        img.src = event.target.result;
      };

      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  imageDataToTensor(imageData) {
    // Simulated tensor conversion
    // In production, this would use WebTorch
    const { width, height, data } = imageData;
    const tensor = {
      shape: [1, 3, height, width], // Batch, Channels, Height, Width
      dtype: "float32",
      data: new Float32Array(width * height * 3),
    };

    // Convert RGBA to RGB and normalize to [0, 1]
    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      tensor.data[j] = data[i] / 255; // R
      tensor.data[j + 1] = data[i + 1] / 255; // G
      tensor.data[j + 2] = data[i + 2] / 255; // B
    }

    return tensor;
  }

  async importText(files) {
    const texts = [];

    for (const file of files) {
      if (
        !this.supportedTextFormats.includes(file.type) &&
        !file.name.match(/\.(txt|json|csv|md)$/i)
      ) {
        console.warn(`Unsupported text format: ${file.name}`);
        continue;
      }

      if (file.size > this.maxTextSize) {
        console.warn(`Text file too large: ${file.name} (${file.size} bytes)`);
        continue;
      }

      try {
        const textData = await this.processText(file);
        texts.push(textData);
      } catch (error) {
        console.error(`Failed to process text ${file.name}:`, error);
      }
    }

    return texts;
  }

  async processText(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (event) => {
        const content = event.target.result;

        // Parse based on file type
        let parsedContent;
        try {
          if (file.type === "application/json" || file.name.endsWith(".json")) {
            parsedContent = JSON.parse(content);
          } else if (file.type === "text/csv" || file.name.endsWith(".csv")) {
            parsedContent = this.parseCSV(content);
          } else {
            parsedContent = content;
          }
        } catch (error) {
          parsedContent = content; // Fallback to raw text
        }

        resolve({
          name: file.name,
          type: file.type,
          size: file.size,
          content: parsedContent,
          timestamp: Date.now(),
          tokens: this.tokenizeText(content),
        });
      };

      reader.onerror = reject;
      reader.readAsText(file);
    });
  }

  parseCSV(content) {
    const lines = content.split("\n");
    const headers = lines[0].split(",").map((h) => h.trim());
    const data = [];

    for (let i = 1; i < lines.length; i++) {
      if (lines[i].trim() === "") continue;

      const values = lines[i].split(",").map((v) => v.trim());
      const row = {};

      for (let j = 0; j < headers.length && j < values.length; j++) {
        row[headers[j]] = values[j];
      }

      data.push(row);
    }

    return data;
  }

  tokenizeText(text) {
    // Simple tokenization for prototype
    const encoder = new TextEncoder();
    const bytes = encoder.encode(text);

    // Pad or truncate to fixed length for NeuralTokenizer (e.g. 128)
    const maxLen = 128;
    const paddedBytes = new Uint8Array(maxLen);
    paddedBytes.set(bytes.slice(0, maxLen));

    return {
      words: text.toLowerCase().match(/\b\w+\b/g) || [],
      characters: text.length,
      sentences: text.split(/[.!?]+/).filter((s) => s.trim() !== ""),
      tokens: Math.ceil(text.length / 4), // Rough estimate
      bytes: Array.from(paddedBytes), // Convert to array for serialization
    };
  }

  async importDataset(files) {
    const images = [];
    const texts = [];

    for (const file of files) {
      if (file.type.startsWith("image/")) {
        try {
          const image = await this.processImage(file);
          images.push(image);
        } catch (error) {
          console.warn(`Failed to import image ${file.name}:`, error);
        }
      } else if (
        file.type.startsWith("text/") ||
        file.name.match(/\.(txt|json|csv|md)$/i)
      ) {
        try {
          const text = await this.processText(file);
          texts.push(text);
        } catch (error) {
          console.warn(`Failed to import text ${file.name}:`, error);
        }
      } else {
        console.warn(`Unsupported file type: ${file.name} (${file.type})`);
      }
    }

    return {
      images,
      texts,
      summary: {
        totalFiles: files.length,
        images: images.length,
        texts: texts.length,
        totalSize: files.reduce((sum, file) => sum + file.size, 0),
      },
    };
  }

  createTrainingBatch(images, batchSize = 4) {
    // Create simulated training batches from imported images
    const batches = [];

    for (let i = 0; i < images.length; i += batchSize) {
      const batchImages = images.slice(i, i + batchSize);
      const batch = {
        images: batchImages,
        labels: this.generateLabels(batchImages.length),
        batchId: `batch_${Date.now()}_${i}`,
        size: batchImages.length,
      };

      batches.push(batch);
    }

    return batches;
  }

  generateLabels(count) {
    // Generate random labels for prototype
    const labels = [];
    for (let i = 0; i < count; i++) {
      labels.push({
        class: Math.floor(Math.random() * 10), // 0-9 for CIFAR/STL-10
        confidence: 0.7 + Math.random() * 0.3,
        source: "imported",
      });
    }
    return labels;
  }

  async saveToDatabase(database, dataset) {
    if (!database) return;

    try {
      // Save images
      for (const image of dataset.images) {
        await database.saveResult({
          type: "IMPORTED_IMAGE",
          name: image.name,
          size: image.size,
          dimensions: `${image.width}x${image.height}`,
          timestamp: image.timestamp,
          data: {
            url: image.url,
            tensorShape: image.tensor.shape,
          },
        });
      }

      // Save texts
      for (const text of dataset.texts) {
        await database.saveResult({
          type: "IMPORTED_TEXT",
          name: text.name,
          size: text.size,
          tokenCount: text.tokens.tokens,
          timestamp: text.timestamp,
          data: {
            preview:
              typeof text.content === "string"
                ? text.content.substring(0, 100) + "..."
                : "[Structured Data]",
          },
        });
      }

      console.log(
        `✅ Saved ${dataset.images.length} images and ${dataset.texts.length} texts to database`,
      );
      return true;
    } catch (error) {
      console.error("Failed to save to database:", error);
      return false;
    }
  }

  getImportStats(dataset) {
    return {
      totalFiles: dataset.summary.totalFiles,
      images: dataset.images.length,
      texts: dataset.texts.length,
      totalSize: this.formatBytes(dataset.summary.totalSize),
      imageStats: {
        totalPixels: dataset.images.reduce(
          (sum, img) => sum + img.width * img.height,
          0,
        ),
        avgDimensions: this.calculateAvgDimensions(dataset.images),
      },
      textStats: {
        totalTokens: dataset.texts.reduce(
          (sum, text) => sum + text.tokens.tokens,
          0,
        ),
        avgTokens:
          dataset.texts.length > 0
            ? dataset.texts.reduce((sum, text) => sum + text.tokens.tokens, 0) /
              dataset.texts.length
            : 0,
      },
    };
  }

  calculateAvgDimensions(images) {
    if (images.length === 0) return "0x0";

    const avgWidth =
      images.reduce((sum, img) => sum + img.width, 0) / images.length;
    const avgHeight =
      images.reduce((sum, img) => sum + img.height, 0) / images.length;

    return `${Math.round(avgWidth)}x${Math.round(avgHeight)}`;
  }

  formatBytes(bytes) {
    if (bytes === 0) return "0 Bytes";

    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));

    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  }

  createDataPreview(dataset, maxPreview = 4) {
    const preview = {
      images: dataset.images.slice(0, maxPreview).map((img) => ({
        name: img.name,
        url: img.url,
        dimensions: `${img.width}x${img.height}`,
        size: this.formatBytes(img.size),
      })),
      texts: dataset.texts.slice(0, maxPreview).map((text) => ({
        name: text.name,
        preview:
          typeof text.content === "string"
            ? text.content.substring(0, 50) + "..."
            : `[${Object.keys(text.content).length} items]`,
        size: this.formatBytes(text.size),
      })),
    };

    return preview;
  }
}

export { DataImporter };
