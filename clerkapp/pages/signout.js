    import { ClerkProvider } from '../../node_modules@clerk/nextjs';
    import { useRouter } from 'next/router';

    function Signout() {
    const router = useRouter();

    const { signOut } = useClerk();

    const handleSignout = async () => {
        await signOut();

        router.push('/');
    };

    return (
        <ClerkProvider>
        <div>
            <h1>Sign out</h1>
            <button onClick={handleSignout}>Sign out</button>
        </div>
        </ClerkProvider>
    );
    }

    export default Signout;