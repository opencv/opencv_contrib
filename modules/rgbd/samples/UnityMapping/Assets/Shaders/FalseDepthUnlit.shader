Shader "Custom/FalseDepthUnlit" {
    Properties{
        _Color("Color", Color) = (1,1,1,1)
        _MainTex("Albedo (RGB)", 2D) = "white" {}
    _Glossiness("Smoothness", Range(0,1)) = 0.5
        _Metallic("Metallic", Range(0,1)) = 0.0
    }
        SubShader{
        Tags{ "RenderType" = "Opaque" }
        LOD 200

        CGPROGRAM
        // Physically based Standard lighting model, and enable shadows on all light types
        //#pragma surface surf Standard fullforwardshadows
#pragma surface surf Unlit vertex:vert

        // Use shader model 3.0 target, to get nicer looking lighting
#pragma target 3.0

        sampler2D _MainTex;
    float4x4 _Homography;

    struct Input {
        float2 uv_MainTex;
        float4 pos;
    };

    half _Glossiness;
    half _Metallic;
    fixed4 _Color;

    void vert(inout appdata_full v, out Input o)
    {
        o.uv_MainTex = v.texcoord.xy;
        o.pos = v.vertex;
    }

    half4 LightingUnlit(SurfaceOutput s, half3 lightDir, half atten)
    {
        return half4(s.Albedo, s.Alpha);
    }

    void surf(Input IN, inout SurfaceOutput o) {
        // Albedo comes from a texture tinted by color
        //fixed4 c = tex2D (_MainTex, IN.uv_MainTex) * _Color;
        float4 newUv = mul(_Homography, IN.pos);
        fixed4 c = tex2D(_MainTex, newUv.xy / newUv.w) * _Color;
        o.Albedo = c.rgb;
        // Metallic and smoothness come from slider variables
        //o.Metallic = _Metallic;
        //o.Smoothness = _Glossiness;
        o.Alpha = c.a;
    }
    ENDCG
    }
        FallBack "Diffuse"
}
