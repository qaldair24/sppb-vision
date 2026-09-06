import time

import cv2


def main():
    camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise SystemExit(
            "No se pudo abrir la camara. En macOS vaya a Ajustes del Sistema > "
            "Privacidad y seguridad > Camara y permita el acceso a Terminal o Python."
        )

    print("Camara detectada. Presione q para cerrar la prueba.")
    limite = time.monotonic() + 30.0
    while time.monotonic() < limite:
        ok, frame = camara.read()
        if not ok:
            raise SystemExit("La camara dejo de entregar imagenes.")
        cv2.putText(
            frame,
            "CAMARA OK - presione q para cerrar",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Prueba de camara 5xSTS-VISION", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camara.release()
    cv2.destroyAllWindows()
    print("Prueba de camara completada.")


if __name__ == "__main__":
    main()
