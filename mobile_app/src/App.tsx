import React, { useState, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  Alert,
  TouchableOpacity,
  Image,
  FlatList,
  ActivityIndicator,
} from "react-native";

import * as ImagePicker from "expo-image-picker";
import * as ImageManipulator from "expo-image-manipulator";
import { loadTensorflowModel } from "react-native-fast-tflite";
import mergedData from "../assets/ver2_embedding_dataset.json";

import jpeg from "jpeg-js"
import { imageMap } from "../assets/DATA_NEW_imageMap";


function base64ToUint8Array(base64: string): Uint8Array {
  const binary = atob(base64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}


interface DatasetVector {
  id: string;
  vector: number[];
}

interface SearchResult {
  id: string;
  similarity: number;
}

export default function App() {
  const [status, setStatus] = useState("🔄 Đang khởi động...");
  const [datasetVectors, setDatasetVectors] = useState<DatasetVector[]>([]);
  const [model, setModel] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const TOP_K_RESULTS = 20;

  useEffect(() => {
    (async () => {
      try {
        console.log("🚀 Bắt đầu tải model...");

        const startTime = performance.now();

        const loadedModel = await loadTensorflowModel(
          require("../assets/ver2_model_float16.tflite")
        );

        const endTime = performance.now();
        const loadTime = endTime - startTime;

        setModel(loadedModel);

        console.log(
          `⏱️ Thời gian khởi động model: ${loadTime.toFixed(2)} ms`
        );

        setDatasetVectors(mergedData as DatasetVector[]);
        setStatus("TẢI THÀNH CÔNG!");

      } catch (error: any) {
        console.error("❌ LỖI:", error);
      }
    })();
  }, []);


  const getImageSize = (uri: string): Promise<{ width: number; height: number }> =>
    new Promise((resolve, reject) => {
      Image.getSize(
        uri,
        (width, height) => resolve({ width, height }),
        (err) => reject(err)
      );
    });

  const processImage = async (uri: string) => {
    try {
      console.log("🖼️ Đang xử lý ảnh:", uri);


      const { width: origW, height: origH } = await getImageSize(uri);
      console.log(`📐 Original size: ${origW}x${origH}`);

      let newW: number, newH: number;
      const shorter = 256;

      if (origW <= origH) {
        newW = shorter;
        newH = Math.round(origH * (shorter / origW));
      } else {
        newH = shorter;
        newW = Math.round(origW * (shorter / origH));
      }

      // Resize ảnh về 256x256 
      const manipulated = await ImageManipulator.manipulateAsync(
        uri,
        [{ resize: { width: newW, height: newH } }],
        { format: ImageManipulator.SaveFormat.JPEG, base64: true }
      );

      console.log(
        `✅ Ảnh đã resize: ${manipulated.width}x${manipulated.height}`
      );

      if (!manipulated.base64) {
        throw new Error("Không nhận được base64 từ ImageManipulator");
      }

      // Decode JPEG → RGBA (256x256)
      const jpegBytes = base64ToUint8Array(manipulated.base64);
      const decoded = jpeg.decode(jpegBytes, { useTArray: true });
      const { width, height, data } = decoded;

      console.log(`✅ Ảnh sau decode: ${width}x${height}`);

      // CenterCrop 224x224 
      const cropSize = 224;
      const startX = Math.floor((width - cropSize) / 2);  // = 16
      const startY = Math.floor((height - cropSize) / 2); // = 16

      const croppedData = new Uint8Array(cropSize * cropSize * 4);

      let idx = 0;
      for (let y = 0; y < cropSize; y++) {
        for (let x = 0; x < cropSize; x++) {
          const srcIdx =
            ((y + startY) * width + (x + startX)) * 4;

          croppedData[idx++] = data[srcIdx];     // R
          croppedData[idx++] = data[srcIdx + 1]; // G
          croppedData[idx++] = data[srcIdx + 2]; // B
          croppedData[idx++] = data[srcIdx + 3]; // A
        }
      }

      console.log("✂️ CenterCrop 224x224 hoàn tất");

      // ToTensor + Normalize (ImageNet)
      const inputTensor = new Float32Array(224 * 224 * 3);
      const mean = [0.485, 0.456, 0.406];
      const std = [0.229, 0.224, 0.225];

      let j = 0;
      for (let i = 0; i < croppedData.length; i += 4) {
        const r = croppedData[i] / 255.0;
        const g = croppedData[i + 1] / 255.0;
        const b = croppedData[i + 2] / 255.0;


        inputTensor[j++] = (r - mean[0]) / std[0];
        inputTensor[j++] = (g - mean[1]) / std[1];
        inputTensor[j++] = (b - mean[2]) / std[2];
      }

      // 5️⃣ Debug nhanh
      let minVal = Infinity;
      let maxVal = -Infinity;
      for (const v of inputTensor) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }

      console.log("📸 Tensor size:", inputTensor.length);
      console.log(`✅ Min: ${minVal.toFixed(4)}, Max: ${maxVal.toFixed(4)}`);
      console.log(
        "🔢 First 10 normalized values:",
        inputTensor.slice(0, 10)
      );

      // 6️⃣ Chạy inference
      await runInference(inputTensor);

    } catch (err) {
      console.error("❌ Lỗi khi xử lý ảnh:", err);
    }
  };



  // Chọn ảnh từ thư viện
  const pickImage = async () => {
    try {

      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });
      if (!result.canceled && result.assets?.length > 0) {
        const uri = result.assets[0].uri;
        setSelectedImage(uri);
        setResults([]);
        await processImage(uri);
      }
    } catch (err) {
      console.error("❌ Lỗi khi chọn ảnh:", err);
    }
  };

  // Chụp ảnh bằng camera
  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== "granted") {
        Alert.alert("Quyền bị từ chối", "Bạn cần cấp quyền camera để chụp ảnh.");
        return;
      }

      const result = await ImagePicker.launchCameraAsync({
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
      });
      if (!result.canceled && result.assets?.length > 0) {
        const uri = result.assets[0].uri;
        setSelectedImage(uri);
        setResults([]);
        await processImage(uri);
      }
    } catch (err) {
      console.error("❌ Lỗi khi chụp ảnh:", err);
    }
  };

  const runInference = async (inputTensor: Float32Array) => {
    if (!model) return;

    setIsProcessing(true);
    try {
      const startInfer = performance.now();

      const output = model.runSync([inputTensor]);
      const out0 = output[0];
      console.log("🧠 output[0] type:", Object.prototype.toString.call(out0));
      console.log("🧠 output[0] length:", (out0 as any)?.length);
      console.log("🧠 output[0] first 10:", Array.from(out0 as any).slice(0, 10));

      const endInfer = performance.now();
      const inferTime = endInfer - startInfer;

      console.log(
        `⚡ Thời gian suy luận (inference): ${inferTime.toFixed(2)} ms`
      );

      const queryVector = output[0] as number[];
      findTopKMatches(queryVector);

    } catch (err) {
      console.error("❌ Lỗi inference:", err);
    } finally {
      setIsProcessing(false);
    }
  };


  const SIMILARITY_THRESHOLD = 0.70;
  const findTopKMatches = (queryVector: number[]) => {
    if (!datasetVectors.length) {
      Alert.alert("Lỗi", "Dữ liệu chưa sẵn sàng.");
      return;
    }


    const similarities: SearchResult[] = datasetVectors.map((item) => ({
      id: item.id,
      similarity: cosineSimilarity(queryVector, item.vector),
    }));


    similarities.sort((a, b) => b.similarity - a.similarity);
    console.log("📦 datasetVectors:", datasetVectors.length);
    console.log("📦 dataset dim:", datasetVectors[0]?.vector?.length);
    console.log("🔎 query dim:", (queryVector as any)?.length);

    const bestMatch = similarities[0];

    if (bestMatch.similarity < SIMILARITY_THRESHOLD) {

      Alert.alert(
        "Không tìm thấy",
        `Sản phẩm này không có trong hệ thống.\n(Độ tin cậy: ${bestMatch.similarity.toFixed(2)})`
      );
      setResults([]);
      return;
    }



    const topK = similarities.slice(0, TOP_K_RESULTS);
    setResults(topK);
  };


  // 7️⃣ Hàm tính Cosine Similarity
  function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dot = 0,
      normA = 0,
      normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dot += vecA[i] * vecB[i];
      normA += vecA[i] ** 2;
      normB += vecB[i] ** 2;
    }
    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom === 0 ? 0 : dot / denom;
  }



  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>CBIR_App</Text>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.actionButton} onPress={pickImage}>
          <Text style={styles.actionButtonIcon}>Chọn ảnh</Text>
          <Text style={styles.actionButtonText}>Từ thư viện</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.actionButton} onPress={takePhoto}>
          <Text style={styles.actionButtonIcon}>Chụp ảnh</Text>
          <Text style={styles.actionButtonText}>Camera</Text>
        </TouchableOpacity>
      </View>

      {selectedImage && (
        <View style={styles.previewCompact}>
          <Text style={styles.previewLabel}>Ảnh của bạn</Text>
          <Image source={{ uri: selectedImage }} style={styles.previewImage} />
        </View>
      )}

      {/* TRẠNG THÁI */}
      <Text style={styles.statusText}>{status}</Text>
      {isProcessing && <ActivityIndicator size="large" color="#007AFF" style={{ marginVertical: 10 }} />}

      {/* KẾT QUẢ */}
      {results.length > 0 && (
        <View style={styles.resultsSection}>
          <Text style={styles.resultsTitle}>
            Kết quả tìm kiếm ({results.length} ảnh)
          </Text>

          {/* Hiển thị ảnh */}
          <FlatList
            data={results}
            keyExtractor={(item) => item.id}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={{ paddingBottom: 30 }}
            renderItem={({ item, index }) => {
              const fileName = item.id.split("\\").pop() || item.id.split("/").pop();
              const imageSource = imageMap[fileName ?? ""] || null;

              return (
                <View style={styles.resultCard}>
                  <Text style={styles.rankBadge}>#{index + 1}</Text>
                  <Image source={imageSource} style={styles.resultImage} resizeMode="cover" />
                  <View style={styles.resultInfo}>
                    <Text style={styles.resultName} numberOfLines={2}>
                      {fileName}
                    </Text>
                    <Text style={styles.similarityText}>
                      {(item.similarity * 100).toFixed(1)}% độ tương đồng
                    </Text>
                  </View>
                </View>
              );
            }}
          />




        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f8f9fa" },


  header: {
    backgroundColor: "#007AFF",
    paddingVertical: 35,
    paddingHorizontal: 20,
    alignItems: "center",
    borderBottomLeftRadius: 30,
    borderBottomRightRadius: 30,
  },
  title: { fontSize: 30, fontWeight: "bold", color: "white" },
  subtitle: { fontSize: 15, color: "#e3f2ff", marginTop: 4, fontWeight: "500" },


  buttonContainer: { flexDirection: "row", justifyContent: "center", gap: 16, marginTop: -30 },
  actionButton: {
    backgroundColor: "white",
    paddingVertical: 5,
    paddingHorizontal: 30,
    borderRadius: 16,
    alignItems: "center",
    minWidth: 120,
    elevation: 8,
  },
  actionButtonIcon: { fontSize: 26 },
  actionButtonText: { fontSize: 13, color: "#333", fontWeight: "600", marginTop: 4 },


  previewCompact: { alignItems: "center", marginTop: 20 },
  previewLabel: { fontSize: 16, fontWeight: "600", color: "#333", marginBottom: 8 },
  previewImage: { width: 180, height: 180, borderRadius: 20, borderWidth: 3, borderColor: "white", elevation: 6 },

  statusText: { marginTop: 12, fontSize: 15, color: "#007AFF", fontWeight: "600", textAlign: "center" },


  resultsSection: { flex: 1, marginTop: 10, paddingHorizontal: 16 },
  resultsTitle: {
    fontSize: 18,
    fontWeight: "bold",
    color: "#1a1a1a",
    marginBottom: 12,
    textAlign: "center",
  },

  resultCard: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "white",
    marginBottom: 14,
    borderRadius: 20,
    padding: 14,
    elevation: 5,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.18,
    shadowRadius: 6,
  },
  rankBadge: {
    position: "absolute",
    top: 10,
    left: 10,
    backgroundColor: "#007AFF",
    color: "white",
    fontWeight: "bold",
    fontSize: 13,
    width: 26,
    height: 26,
    textAlign: "center",
    lineHeight: 26,
    borderRadius: 13,
    zIndex: 10,
  },
  resultImage: { width: 86, height: 86, borderRadius: 14, marginRight: 16 },
  resultInfo: { flex: 1 },
  resultName: { fontSize: 14, color: "#333", fontWeight: "600", marginBottom: 6 },
  similarityText: { fontSize: 18, fontWeight: "bold", color: "#007AFF" },



  simpleResultCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'white',
    marginBottom: 14,
    padding: 16,
    borderRadius: 20,
    elevation: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.15,
    shadowRadius: 8,
  },
  rankCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  rankNumber: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
  },
  simpleResultInfo: {
    flex: 1,
  },
  simpleFileName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#2d3436',
    marginBottom: 6,
  },
  simpleSimilarity: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#5E60CE',
  },
});
