from imu_lumbar import ErrorIMU, probar_conexion


try:
    probar_conexion()
except ErrorIMU as exc:
    print(f"\nERROR: {exc}")
    input("\nPresione Enter para cerrar...")
    raise SystemExit(1)

input("\nPresione Enter para cerrar...")
