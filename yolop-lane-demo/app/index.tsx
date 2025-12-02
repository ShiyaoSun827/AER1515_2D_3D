import React, { useEffect, useRef, useState } from "react";
import { StyleSheet, View, Image, Button, Text } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";

const SERVER_URL = "http://192.168.0.10:8000/infer"; // 换成你的 Mac IP

export default function HomeScreen() {
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);

  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  const lastTimeRef = useRef<number | null>(null); // 用来算实际 FPS（可选）

  // 请求权限
  useEffect(() => {
    if (!permission || !permission.granted) {
      requestPermission();
    }
  }, [permission]);

  // ⭐ 关键：推理循环，不用 setInterval
  useEffect(() => {
    let cancelled = false;

    const loop = async () => {
      if (cancelled || !processing) return;
      await captureAndSend();          // 等上一帧推理结束
      if (!cancelled && processing) {
        requestAnimationFrame(loop);   // 尽快请求下一帧
      }
    };

    if (processing) {
      loop();
    }

    return () => {
      cancelled = true;
    };
  }, [processing]);

  const captureAndSend = async () => {
    if (!cameraRef.current) return;

    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.1,       // 再小一点会更快
        skipProcessing: true,
      });

      if (!photo?.base64) return;

      const t0 = Date.now();

      const response = await fetch(SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: photo.base64 }),
      });

      const json = await response.json();
      if (json.image_base64) {
        setSegmentedImage(json.image_base64);
      }

      // 打印一下当前“检测帧率”，大概有个概念
      if (lastTimeRef.current !== null) {
        const dt = (t0 - lastTimeRef.current) / 1000;
        const fps = 1 / dt;
        console.log("effective detection FPS:", fps.toFixed(2));
      }
      lastTimeRef.current = t0;
    } catch (err) {
      console.warn("infer error", err);
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.center}>
        <Text>请求摄像头权限中…</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* 隐藏的 CameraView，只拿帧用 */}
      <CameraView
        ref={cameraRef}
        style={{ width: 1, height: 1, opacity: 0 }}
        facing={"back"}
      />

      {/* 全屏显示推理后的结果 */}
      {segmentedImage ? (
        <Image
          style={[StyleSheet.absoluteFillObject, { opacity: 0.9 }]}
          source={{ uri: "data:image/jpeg;base64," + segmentedImage }}
        />
      ) : (
        <View style={styles.center}>
          <Text>等待第一帧分割结果…</Text>
        </View>
      )}

      <View style={styles.bottomBar}>
        <Button
          title={processing ? "STOP DETECTION & SEGMENTATION" : "START DETECTION & SEGMENTATION"}
          onPress={() => setProcessing((prev) => !prev)}
        />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },
  bottomBar: {
    position: "absolute",
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});
