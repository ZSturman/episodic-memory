using UnityEngine;

namespace EpisodicAgent.Player
{
    /// <summary>
    /// Simple first-person controller with WASD + mouse look.
    /// Designed for the sensor simulation environment.
    /// </summary>
    [RequireComponent(typeof(CharacterController))]
    public class FirstPersonController : MonoBehaviour
    {
        [Header("Movement")]
        [SerializeField] private float walkSpeed = 5f;
        [SerializeField] private float runSpeed = 8f;
        [SerializeField] private float gravity = -9.81f;
        [SerializeField] private float jumpHeight = 1.5f;

        [Header("Mouse Look")]
        [SerializeField] private float mouseSensitivity = 2f;
        [SerializeField] private float maxLookAngle = 85f;
        [SerializeField] private Transform cameraTransform;

        [Header("Ground Check")]
        [SerializeField] private Transform groundCheck;
        [SerializeField] private float groundDistance = 0.4f;
        [SerializeField] private LayerMask groundMask = -1;  // Everything by default

        private CharacterController controller;
        private Vector3 velocity;
        private bool isGrounded;
        private float verticalRotation;
        private bool cursorLocked = true;

        private void Awake()
        {
            controller = GetComponent<CharacterController>();

            // Auto-find camera if not set
            if (cameraTransform == null)
            {
                Camera cam = GetComponentInChildren<Camera>();
                if (cam != null)
                {
                    cameraTransform = cam.transform;
                }
                else
                {
                    cameraTransform = Camera.main?.transform;
                }
            }

            // Create ground check if not set
            if (groundCheck == null)
            {
                GameObject go = new GameObject("GroundCheck");
                go.transform.SetParent(transform);
                go.transform.localPosition = new Vector3(0, -1f, 0);
                groundCheck = go.transform;
            }
        }

        private void Start()
        {
            LockCursor();
        }

        private void Update()
        {
            HandleCursorLock();
            
            if (cursorLocked)
            {
                HandleMouseLook();
            }
            
            HandleMovement();
        }

        private void HandleCursorLock()
        {
            // Toggle cursor lock with Escape
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                if (cursorLocked)
                {
                    UnlockCursor();
                }
                else
                {
                    LockCursor();
                }
            }

            // Re-lock on click
            if (!cursorLocked && Input.GetMouseButtonDown(0))
            {
                LockCursor();
            }
        }

        private void LockCursor()
        {
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
            cursorLocked = true;
        }

        private void UnlockCursor()
        {
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            cursorLocked = false;
        }

        private void HandleMouseLook()
        {
            float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
            float mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;

            // Horizontal rotation - rotate the whole character
            transform.Rotate(Vector3.up * mouseX);

            // Vertical rotation - only rotate the camera
            verticalRotation -= mouseY;
            verticalRotation = Mathf.Clamp(verticalRotation, -maxLookAngle, maxLookAngle);

            if (cameraTransform != null)
            {
                cameraTransform.localRotation = Quaternion.Euler(verticalRotation, 0, 0);
            }
        }

        private void HandleMovement()
        {
            // Ground check
            isGrounded = Physics.CheckSphere(groundCheck.position, groundDistance, groundMask);

            if (isGrounded && velocity.y < 0)
            {
                velocity.y = -2f;  // Small downward force to keep grounded
            }

            // Get input
            float horizontal = Input.GetAxis("Horizontal");
            float vertical = Input.GetAxis("Vertical");

            // Calculate move direction relative to player orientation
            Vector3 move = transform.right * horizontal + transform.forward * vertical;

            // Determine speed
            float speed = Input.GetKey(KeyCode.LeftShift) ? runSpeed : walkSpeed;

            // Apply movement
            controller.Move(move * speed * Time.deltaTime);

            // Jump
            if (Input.GetButtonDown("Jump") && isGrounded)
            {
                velocity.y = Mathf.Sqrt(jumpHeight * -2f * gravity);
            }

            // Apply gravity
            velocity.y += gravity * Time.deltaTime;
            controller.Move(velocity * Time.deltaTime);
        }

        /// <summary>
        /// Teleport player to a position.
        /// </summary>
        public void Teleport(Vector3 position)
        {
            controller.enabled = false;
            transform.position = position;
            velocity = Vector3.zero;
            controller.enabled = true;
        }

        /// <summary>
        /// Teleport player to a position with a specific rotation.
        /// </summary>
        public void Teleport(Vector3 position, Quaternion rotation)
        {
            controller.enabled = false;
            transform.position = position;
            transform.rotation = Quaternion.Euler(0, rotation.eulerAngles.y, 0);
            verticalRotation = 0;
            if (cameraTransform != null)
            {
                cameraTransform.localRotation = Quaternion.identity;
            }
            velocity = Vector3.zero;
            controller.enabled = true;
        }

        /// <summary>
        /// Get current movement velocity (for sensor data).
        /// </summary>
        public Vector3 GetVelocity()
        {
            return controller.velocity;
        }

        /// <summary>
        /// Check if player is currently grounded.
        /// </summary>
        public bool IsGrounded()
        {
            return isGrounded;
        }
    }
}
