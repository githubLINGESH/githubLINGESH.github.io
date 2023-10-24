    import { useClerk } from '../../node_modules@clerk/nextjs';

    function MyPage() {
    const { user } = useClerk();

    if (!user) {
        return <div>Please sign in.</div>;
    }

    return <div>Hello, {user.email}!</div>;
    }

    export default MyPage;
