"""
Script para visualizar y gestionar sesiones guardadas en MongoDB
"""
from pymongo import MongoClient
from datetime import datetime
import json
from bson import ObjectId

# Configuración
MONGO_URI = "mongodb://admin:password@localhost:27017/"
client = MongoClient(MONGO_URI)
db = client['facesense_db']
sessions_collection = db['therapy_sessions']


def list_all_sessions():
    """Lista todas las sesiones guardadas"""
    sessions = list(sessions_collection.find().sort('start_time', -1))
    
    print("\n" + "="*70)
    print("🗂️  SESIONES GUARDADAS")
    print("="*70)
    
    if not sessions:
        print("❌ No hay sesiones guardadas")
        return
    
    for i, session in enumerate(sessions, 1):
        print(f"\n📁 Sesión #{i}")
        print(f"   ID: {session['_id']}")
        print(f"   Session ID: {session.get('session_id', 'N/A')}")
        print(f"   Inicio: {session.get('start_time', 'N/A')}")
        print(f"   Estado: {session.get('status', 'N/A')}")
        print(f"   Video: {session.get('video_path', 'N/A')}")
        print(f"   Interacciones: {len(session.get('interactions', []))}")


def view_session_details(session_id):
    """Muestra detalles completos de una sesión"""
    try:
        session = sessions_collection.find_one({'_id': ObjectId(session_id)})
        
        if not session:
            print(f"❌ Sesión {session_id} no encontrada")
            return
        
        print("\n" + "="*70)
        print(f"📊 DETALLES DE SESIÓN: {session_id}")
        print("="*70)
        
        print(f"\n🆔 Session ID: {session.get('session_id')}")
        print(f"🕐 Inicio: {session.get('start_time')}")
        print(f"🕑 Fin: {session.get('end_time', 'En curso')}")
        print(f"📹 Video: {session.get('video_path', 'N/A')}")
        print(f"📊 Estado: {session.get('status')}")
        
        interactions = session.get('interactions', [])
        print(f"\n💬 Interacciones: {len(interactions)}")
        
        for i, interaction in enumerate(interactions, 1):
            print(f"\n--- Interacción #{i} ---")
            print(f"⏰ Timestamp: {interaction.get('timestamp')}")
            print(f"👤 Usuario: {interaction.get('user_message')}")
            print(f"🐱 Terapeuta: {interaction.get('therapist_response')}")
            
            # Emociones
            fusion = interaction.get('fusion_result', {})
            print(f"😊 Emoción Principal: {fusion.get('emocion_principal', 'N/A')} "
                  f"({fusion.get('confianza_principal', 0):.1f}%)")
            
            if fusion.get('hay_conflicto'):
                print(f"⚠️  Conflicto Emocional:")
                print(f"   Texto: {fusion.get('emocion_texto_dominante')}")
                print(f"   Rostro: {fusion.get('emocion_rostro_dominante')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def export_session_json(session_id, output_file='session_export.json'):
    """Exporta una sesión a JSON"""
    try:
        session = sessions_collection.find_one({'_id': ObjectId(session_id)})
        
        if not session:
            print(f"❌ Sesión {session_id} no encontrada")
            return
        
        # Convertir ObjectId a string para JSON
        session['_id'] = str(session['_id'])
        
        # Convertir datetime a string
        if 'start_time' in session:
            session['start_time'] = session['start_time'].isoformat()
        if 'end_time' in session:
            session['end_time'] = session['end_time'].isoformat()
        
        for interaction in session.get('interactions', []):
            if 'timestamp' in interaction:
                interaction['timestamp'] = interaction['timestamp'].isoformat()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(session, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Sesión exportada a: {output_file}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def get_emotion_stats():
    """Estadísticas generales de emociones detectadas"""
    sessions = list(sessions_collection.find())
    
    emotion_counts = {}
    total_interactions = 0
    
    for session in sessions:
        for interaction in session.get('interactions', []):
            total_interactions += 1
            fusion = interaction.get('fusion_result', {})
            emotion = fusion.get('emocion_principal', 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("\n" + "="*70)
    print("📊 ESTADÍSTICAS GENERALES")
    print("="*70)
    print(f"Total de sesiones: {len(sessions)}")
    print(f"Total de interacciones: {total_interactions}")
    print("\nDistribución de emociones:")
    
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_interactions * 100) if total_interactions > 0 else 0
        print(f"  {emotion}: {count} ({percentage:.1f}%)")


def delete_session(session_id):
    """Elimina una sesión"""
    try:
        result = sessions_collection.delete_one({'_id': ObjectId(session_id)})
        
        if result.deleted_count > 0:
            print(f"✅ Sesión {session_id} eliminada")
        else:
            print(f"❌ Sesión {session_id} no encontrada")
    
    except Exception as e:
        print(f"❌ Error: {e}")


def menu():
    """Menú interactivo"""
    while True:
        print("\n" + "="*70)
        print("🐱 FACESENSE - GESTOR DE SESIONES MONGODB")
        print("="*70)
        print("1. Listar todas las sesiones")
        print("2. Ver detalles de una sesión")
        print("3. Exportar sesión a JSON")
        print("4. Ver estadísticas generales")
        print("5. Eliminar una sesión")
        print("0. Salir")
        
        choice = input("\nOpción: ").strip()
        
        if choice == '1':
            list_all_sessions()
        
        elif choice == '2':
            session_id = input("ID de sesión (ObjectId): ").strip()
            view_session_details(session_id)
        
        elif choice == '3':
            session_id = input("ID de sesión (ObjectId): ").strip()
            output = input("Archivo de salida (default: session_export.json): ").strip()
            if not output:
                output = 'session_export.json'
            export_session_json(session_id, output)
        
        elif choice == '4':
            get_emotion_stats()
        
        elif choice == '5':
            session_id = input("ID de sesión a eliminar (ObjectId): ").strip()
            confirm = input(f"¿Seguro que deseas eliminar {session_id}? (si/no): ").strip().lower()
            if confirm == 'si':
                delete_session(session_id)
        
        elif choice == '0':
            print("👋 Hasta luego!")
            break
        
        else:
            print("❌ Opción inválida")


if __name__ == "__main__":
    try:
        # Test de conexión
        client.server_info()
        print("✅ Conectado a MongoDB")
        
        menu()
    
    except Exception as e:
        print(f"❌ Error de conexión a MongoDB: {e}")
        print("Verifica que Docker esté ejecutando MongoDB")