import React, { useEffect, useRef, useState } from "react";
import { 
  StyleSheet, View, Image, Text, TextInput, TouchableOpacity, 
  KeyboardAvoidingView, Platform, Button, Keyboard // <--- [1] 引入 Keyboard
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ScreenOrientation from 'expo-screen-orientation'; 
import { DeviceMotion } from 'expo-sensors'; 

// REPLACE WITH YOUR ACTUAL SERVER IP
const SERVER_URL = "http://192.168.0.10:8000/infer"; 

export default function HomeScreen() {
  const [segmentedImage, setSegmentedImage] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView | null>(null);
  
  // --- Geometric & FCW Parameters ---
  const [camHeight, setCamHeight] = useState("1.3");     
  const [fcwThreshold, setFcwThreshold] = useState("15"); 
  const [pitch, setPitch] = useState(0.0);               
  const [fps, setFps] = useState("0");
  
  const lastTimeRef = useRef<number | null>(null);

  useEffect(() => {
    async function setup() {
      try {
        await ScreenOrientation.lockAsync(ScreenOrientation.OrientationLock.PORTRAIT_UP);
      } catch (e) {
        console.log("Orientation lock failed", e);
      }
      if (!permission || !permission.granted) {
        requestPermission();
      }
    }
    setup();
  }, [permission]);

  useEffect(() => {
    DeviceMotion.setUpdateInterval(100);
    const subscription = DeviceMotion.addListener((data) => {
      if (data.rotation) {
        setPitch(data.rotation.beta - 1.5708); 
      }
    });
    return () => subscription.remove();
  }, []);

  useEffect(() => {
    let cancelled = false;
    const loop = async () => {
      if (cancelled || !processing) return;
      await captureAndSend();
      if (!cancelled && processing) {
        requestAnimationFrame(loop);
      }
    };
    if (processing) loop();
    return () => { cancelled = true; };
  }, [processing]);

  const captureAndSend = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.3, 
        skipProcessing: true, 
      });

      if (!photo?.base64) return;
      const t0 = Date.now();

      const bodyData = {
        image_base64: photo.base64,
        cam_height: parseFloat(camHeight) || 1.3,
        pitch: pitch, 
        fov_v: 60.0, 
        fcw_threshold: parseFloat(fcwThreshold) || 15.0 
      };

      const response = await fetch(SERVER_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(bodyData),
      });

      const json = await response.json();
      if (json.image_base64) {
        setSegmentedImage(json.image_base64);
      }

      if (lastTimeRef.current !== null) {
        const dt = (t0 - lastTimeRef.current) / 1000;
        setFps((1 / dt).toFixed(1));
      }
      lastTimeRef.current = t0;

    } catch (err) {
      console.warn("Infer error:", err);
      setProcessing(false); 
    }
  };

  if (!permission?.granted) {
    return (
      <View style={styles.center}>
        <Text style={{color: 'white', marginBottom: 20}}>Camera permission required.</Text>
        <Button title="Grant Permission" onPress={requestPermission} />
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={{ width: 1, height: 1, opacity: 0 }}
        facing={"back"}
      />

      {segmentedImage ? (
        <Image
          style={StyleSheet.absoluteFill}
          source={{ uri: "data:image/jpeg;base64," + segmentedImage }}
          resizeMode="contain" 
        />
      ) : (
        <View style={styles.center}>
          <Text style={{ color: '#888', fontSize: 16 }}>AI System Standby...</Text>
        </View>
      )}

      {/* --- HUD Overlay --- */}
      <View style={styles.horizonLine}>
        <Text style={styles.horizonText}>▲ Align with Horizon (Pitch ~0°) ▲</Text>
      </View>

      <View style={styles.hudContainer}>
        <Text style={styles.hudText}>FPS: {fps}</Text>
        <Text style={styles.hudText}>Pitch: {(pitch * 57.3).toFixed(1)}°</Text>
      </View>

      {/* --- Bottom Controls --- */}
      <KeyboardAvoidingView 
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        style={styles.bottomWrapper}
      >
        <View style={styles.controlPanel}>
          {/* Config Inputs */}
          <View style={styles.configRow}>
            <View style={styles.inputGroup}>
              <Text style={styles.label}>Height (m)</Text>
              <TextInput
                style={styles.input}
                value={camHeight}
                onChangeText={setCamHeight}
                keyboardType="numeric"
                returnKeyType="done" // 尝试添加 returnKey
                placeholder="1.3"
                placeholderTextColor="#666"
              />
            </View>

            <View style={styles.inputGroup}>
              <Text style={[styles.label, {color: '#ff4444'}]}>Warn Dist (m)</Text>
              <TextInput
                style={[styles.input, {borderColor: '#ff4444', color: '#ff4444'}]}
                value={fcwThreshold}
                onChangeText={setFcwThreshold}
                keyboardType="numeric"
                returnKeyType="done"
                placeholder="15"
                placeholderTextColor="#666"
              />
            </View>
          </View>

          {/* [新增] 确认/收起键盘按钮 */}
          <TouchableOpacity 
            style={styles.dismissButton} 
            onPress={() => Keyboard.dismiss()} 
          >
            <Text style={styles.dismissButtonText}>▼ OK / Close Keyboard</Text>
          </TouchableOpacity>

          {/* Start/Stop Button */}
          <TouchableOpacity
            style={[
              styles.button, 
              { backgroundColor: processing ? '#ff4444' : '#007AFF' }
            ]}
            onPress={() => {
              Keyboard.dismiss(); // 点击开始时也顺便收起键盘
              setProcessing(!processing);
            }}
          >
            <Text style={styles.buttonText}>
              {processing ? "STOP FCW" : "START FCW"}
            </Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { 
    flex: 1, 
    backgroundColor: "#111", 
  },
  center: { 
    flex: 1, 
    justifyContent: "center", 
    alignItems: "center", 
  },
  horizonLine: {
    position: 'absolute',
    top: '50%', left: 0, right: 0, height: 1,
    backgroundColor: 'rgba(0, 255, 0, 0.5)', 
    justifyContent: 'flex-end', alignItems: 'center',
    zIndex: 10,
  },
  horizonText: {
    color: 'rgba(0, 255, 0, 0.7)', 
    fontSize: 10, marginBottom: 2
  },
  hudContainer: {
    position: 'absolute', top: 50, left: 20, zIndex: 20,
  },
  hudText: {
    color: '#00ff00', fontSize: 14, fontFamily: 'monospace',
    textShadowColor: 'black', textShadowRadius: 2, marginBottom: 2
  },
  bottomWrapper: {
    position: 'absolute', bottom: 0, left: 0, right: 0, zIndex: 30,
  },
  controlPanel: {
    backgroundColor: 'rgba(20, 20, 20, 0.95)', // 稍微加深背景
    padding: 20,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    alignItems: 'center',
  },
  configRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    marginBottom: 10, // 稍微减少间距
  },
  inputGroup: {
    alignItems: 'center',
  },
  label: {
    color: '#ccc', fontSize: 12, marginBottom: 5, fontWeight: '600',
  },
  input: {
    backgroundColor: 'rgba(255,255,255,0.1)',
    color: 'white',
    width: 90, height: 45, // 稍微加大点击区域
    borderRadius: 8,
    textAlign: 'center',
    fontSize: 18,
    borderWidth: 1,
    borderColor: '#555',
  },
  // [新增] 收起键盘按钮样式
  dismissButton: {
    width: '100%',
    paddingVertical: 8,
    marginBottom: 10,
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    borderRadius: 5,
  },
  dismissButtonText: {
    color: '#aaa',
    fontSize: 12,
  },
  button: {
    width: '100%',
    paddingVertical: 15,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white', fontSize: 18, fontWeight: 'bold', letterSpacing: 1,
  },
});